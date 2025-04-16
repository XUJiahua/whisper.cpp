#include "common.h"
#include "sense-voice.h"
#include "httplib.h"
#include "json.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

using namespace httplib;
using json = nlohmann::ordered_json;

// output formats
const std::string json_format   = "json";
const std::string text_format   = "text";
const std::string srt_format    = "srt";
const std::string vjson_format  = "verbose_json";
const std::string vtt_format    = "vtt";

struct server_params
{
    std::string hostname = "127.0.0.1";
    std::string public_path = "examples/server/public";
    std::string request_path = "";
    std::string inference_path = "/inference";

    int32_t port          = 0;
    int32_t read_timeout  = 600;
    int32_t write_timeout = 600;
};

// command-line parameters
struct sense_voice_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t max_len       = 0;
    int32_t n_mel       = 80;
    int32_t best_of       = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    int32_t audio_ctx     = 0;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float grammar_penalty = 100.0f;
    float temperature     = 0.0f;
    float temperature_inc = 0.2f;

    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool output_txt      = false;
    bool output_vtt      = false;
    bool output_srt      = false;
    bool output_wts      = false;
    bool output_csv      = false;
    bool output_jsn      = false;
    bool output_jsn_full = false;
    bool output_lrc      = false;
    bool no_prints       = false;
    bool print_special   = false;
    bool print_colors    = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool log_score       = false;
    bool use_gpu         = true;
    bool flash_attn      = false;
    bool use_itn         = false;

    std::string language  = "auto";
    std::string prompt;
    std::string model     = "models/ggml-base.en.bin";

    std::string response_format     = json_format;


    std::string openvino_encode_device = "CPU";


    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};

};

static int sense_voice_has_coreml(void) {
#ifdef SENSE_VOICE_USE_COREML
    return 1;
#else
    return 0;
#endif
}

static int sense_voice_has_openvino(void) {
#ifdef SENSE_VOICE_USE_OPENVINO
    return 1;
#else
    return 0;
#endif
}

const char * sense_voice_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "SSSE3 = "     + std::to_string(ggml_cpu_has_ssse3())     + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";
    s += "COREML = "    + std::to_string(sense_voice_has_coreml()) + " | ";
    s += "OPENVINO = "  + std::to_string(sense_voice_has_openvino());

    return s.c_str();
}

static void sense_voice_print_usage(int /*argc*/, char ** argv, const sense_voice_params & params, const server_params & sparams) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] file0.wav file1.wav ...\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N         [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "  -p N,      --processors N      [%-7d] number of processors to use during computation\n", params.n_processors);
    fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n",                    params.offset_t_ms);
    fprintf(stderr, "  -on N,     --offset-n N        [%-7d] segment index offset\n",                           params.offset_n);
    fprintf(stderr, "  -d  N,     --duration N        [%-7d] duration of audio to process in milliseconds\n",   params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N     [%-7d] maximum number of text context tokens to store\n", params.max_context);
    fprintf(stderr, "  -ml N,     --max-len N         [%-7d] maximum segment length in characters\n",           params.max_len);
    fprintf(stderr, "  -sow,      --split-on-word     [%-7s] split on word rather than on token\n",             params.split_on_word ? "true" : "false");
    fprintf(stderr, "  -bo N,     --best-of N         [%-7d] number of best candidates to keep\n",              params.best_of);
    fprintf(stderr, "  -bs N,     --beam-size N       [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -ac N,     --audio-ctx N       [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -wt N,     --word-thold N      [%-7.2f] word timestamp probability threshold\n",         params.word_thold);
    fprintf(stderr, "  -et N,     --entropy-thold N   [%-7.2f] entropy threshold for decoder fail\n",           params.entropy_thold);
    fprintf(stderr, "  -lpt N,    --logprob-thold N   [%-7.2f] log probability threshold for decoder fail\n",   params.logprob_thold);
    fprintf(stderr, "  -tp,       --temperature N     [%-7.2f] The sampling temperature, between 0 and 1\n",    params.temperature);
    fprintf(stderr, "  -tpi,      --temperature-inc N [%-7.2f] The increment of temperature, between 0 and 1\n",params.temperature_inc);
    fprintf(stderr, "  -debug,    --debug-mode        [%-7s] enable debug mode (eg. dump log_mel)\n",           params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -di,       --diarize           [%-7s] stereo audio diarization\n",                       params.diarize ? "true" : "false");
    fprintf(stderr, "  -tdrz,     --tinydiarize       [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -nf,       --no-fallback       [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -otxt,     --output-txt        [%-7s] output result in a text file\n",                   params.output_txt ? "true" : "false");
    fprintf(stderr, "  -ovtt,     --output-vtt        [%-7s] output result in a vtt file\n",                    params.output_vtt ? "true" : "false");
    fprintf(stderr, "  -osrt,     --output-srt        [%-7s] output result in a srt file\n",                    params.output_srt ? "true" : "false");
    fprintf(stderr, "  -olrc,     --output-lrc        [%-7s] output result in a lrc file\n",                    params.output_lrc ? "true" : "false");
    fprintf(stderr, "  -owts,     --output-words      [%-7s] output script for generating karaoke video\n",     params.output_wts ? "true" : "false");
    fprintf(stderr, "  -ocsv,     --output-csv        [%-7s] output result in a CSV file\n",                    params.output_csv ? "true" : "false");
    fprintf(stderr, "  -oj,       --output-json       [%-7s] output result in a JSON file\n",                   params.output_jsn ? "true" : "false");
    fprintf(stderr, "  -ojf,      --output-json-full  [%-7s] include more information in the JSON file\n",      params.output_jsn_full ? "true" : "false");
    fprintf(stderr, "  -of FNAME, --output-file FNAME [%-7s] output file path (without file extension)\n",      "");
    fprintf(stderr, "  -np,       --no-prints         [%-7s] do not print anything other than the results\n",   params.no_prints ? "true" : "false");
    fprintf(stderr, "  -ps,       --print-special     [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -pc,       --print-colors      [%-7s] print colors\n",                                   params.print_colors ? "true" : "false");
    fprintf(stderr, "  -pp,       --print-progress    [%-7s] print progress\n",                                 params.print_progress ? "true" : "false");
    fprintf(stderr, "  -nt,       --no-timestamps     [%-7s] do not print timestamps\n",                        params.no_timestamps ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for auto-detect), support [`zh`, `en`, `yue`, `ja`, `ko`\n", params.language.c_str());
    fprintf(stderr, "             --prompt PROMPT     [%-7s] initial prompt (max n_text_ctx/2 tokens)\n",       params.prompt.c_str());
    fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME,  --file FNAME        [%-7s] input WAV file path\n",                            "");
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n",  params.openvino_encode_device.c_str());
    fprintf(stderr, "  -ls,       --log-score         [%-7s] log best decoder scores of tokens\n",              params.log_score?"true":"false");
    fprintf(stderr, "  -ng,       --no-gpu            [%-7s] disable GPU\n",                                    params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,       --flash-attn        [%-7s] flash attention\n",                                params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -itn,      --use-itn           [%-7s] use itn\n",                                        params.use_itn ? "true" : "false");
    fprintf(stderr, "\n");
    // server params
    fprintf(stderr, "  --host HOST,                   [%-7s] Hostname/ip-adress for the server\n", sparams.hostname.c_str());
    fprintf(stderr, "  --port PORT,                   [%-7d] Port number for the server\n", sparams.port);
    fprintf(stderr, "  --public PATH,                 [%-7s] Path to the public folder\n", sparams.public_path.c_str());
    fprintf(stderr, "  --request-path PATH,           [%-7s] Request path for all requests\n", sparams.request_path.c_str());
    fprintf(stderr, "  --inference-path PATH,         [%-7s] Inference path for all requests\n", sparams.inference_path.c_str());
}

struct sense_voice_print_user_data {
    const sense_voice_params * params;

    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

static char * sense_voice_param_turn_lowercase(char * in){
    int string_len = strlen(in);
    for (int i = 0; i < string_len; i++){
        *(in+i) = tolower((unsigned char)*(in+i));
    }
    return in;
}

static bool sense_voice_params_parse(int argc, char ** argv, sense_voice_params & params, server_params & sparams) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-"){
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg[0] != '-') {
            params.fname_inp.push_back(arg);
            continue;
        }

        if (arg == "-h" || arg == "--help") {
            sense_voice_print_usage(argc, argv, params, sparams);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")         { params.n_threads       = std::stoi(argv[++i]); }
        else if (arg == "-p"    || arg == "--processors")      { params.n_processors    = std::stoi(argv[++i]); }
        else if (arg == "-ot"   || arg == "--offset-t")        { params.offset_t_ms     = std::stoi(argv[++i]); }
        else if (arg == "-on"   || arg == "--offset-n")        { params.offset_n        = std::stoi(argv[++i]); }
        else if (arg == "-d"    || arg == "--duration")        { params.duration_ms     = std::stoi(argv[++i]); }
        else if (arg == "-mc"   || arg == "--max-context")     { params.max_context     = std::stoi(argv[++i]); }
        else if (arg == "-ml"   || arg == "--max-len")         { params.max_len         = std::stoi(argv[++i]); }
        else if (arg == "-bo"   || arg == "--best-of")         { params.best_of         = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")       { params.beam_size       = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")       { params.audio_ctx       = std::stoi(argv[++i]); }
        else if (arg == "-wt"   || arg == "--word-thold")      { params.word_thold      = std::stof(argv[++i]); }
        else if (arg == "-et"   || arg == "--entropy-thold")   { params.entropy_thold   = std::stof(argv[++i]); }
        else if (arg == "-lpt"  || arg == "--logprob-thold")   { params.logprob_thold   = std::stof(argv[++i]); }
        else if (arg == "-tp"   || arg == "--temperature")     { params.temperature     = std::stof(argv[++i]); }
        else if (arg == "-tpi"  || arg == "--temperature-inc") { params.temperature_inc = std::stof(argv[++i]); }
        else if (arg == "-debug"|| arg == "--debug-mode")      { params.debug_mode      = true; }
        else if (arg == "-tr"   || arg == "--translate")       { params.translate       = true; }
        else if (arg == "-di"   || arg == "--diarize")         { params.diarize         = true; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")     { params.tinydiarize     = true; }
        else if (arg == "-sow"  || arg == "--split-on-word")   { params.split_on_word   = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")     { params.no_fallback     = true; }
        else if (arg == "-otxt" || arg == "--output-txt")      { params.output_txt      = true; }
        else if (arg == "-ovtt" || arg == "--output-vtt")      { params.output_vtt      = true; }
        else if (arg == "-osrt" || arg == "--output-srt")      { params.output_srt      = true; }
        else if (arg == "-owts" || arg == "--output-words")    { params.output_wts      = true; }
        else if (arg == "-olrc" || arg == "--output-lrc")      { params.output_lrc      = true; }
        else if (arg == "-ocsv" || arg == "--output-csv")      { params.output_csv      = true; }
        else if (arg == "-oj"   || arg == "--output-json")     { params.output_jsn      = true; }
        else if (arg == "-ojf"  || arg == "--output-json-full"){ params.output_jsn_full = params.output_jsn = true; }
        else if (arg == "-of"   || arg == "--output-file")     { params.fname_out.emplace_back(argv[++i]); }
        else if (arg == "-np"   || arg == "--no-prints")       { params.no_prints       = false; }
        else if (arg == "-ps"   || arg == "--print-special")   { params.print_special   = true; }
        else if (arg == "-pc"   || arg == "--print-colors")    { params.print_colors    = true; }
        else if (arg == "-pp"   || arg == "--print-progress")  { params.print_progress  = true; }
        else if (arg == "-nt"   || arg == "--no-timestamps")   { params.no_timestamps   = true; }
        else if (arg == "-l"    || arg == "--language")        { params.language        = sense_voice_param_turn_lowercase(argv[++i]); }
        else if (arg == "-dl"   || arg == "--detect-language") { params.detect_language = true; }
        else if (                  arg == "--prompt")          { params.prompt          = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")           { params.model           = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")            { params.fname_inp.emplace_back(argv[++i]); }
        else if (arg == "-oved" || arg == "--ov-e-device")     { params.openvino_encode_device = argv[++i]; }
        else if (arg == "-ls"   || arg == "--log-score")       { params.log_score       = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")          { params.use_gpu         = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")      { params.flash_attn      = true; }
        else if (                  arg == "--grammar-penalty") { params.grammar_penalty = std::stof(argv[++i]); }
        else if (arg == "-itn"  || arg == "--use-itn")         { params.use_itn         = true; }
        // server params
        else if (                  arg == "--port")            { sparams.port        = std::stoi(argv[++i]); }
        else if (                  arg == "--host")            { sparams.hostname    = argv[++i]; }
        else if (                  arg == "--public")          { sparams.public_path = argv[++i]; }
        else if (                  arg == "--request-path")    { sparams.request_path = argv[++i]; }
        else if (                  arg == "--inference-path")  { sparams.inference_path = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            sense_voice_print_usage(argc, argv, params, sparams);
            exit(0);
        }
    }

    return true;
}

bool parse_str_to_bool(const std::string & s) {
    if (s == "true" || s == "1" || s == "yes" || s == "y") {
        return true;
    }
    return false;
}

void get_req_parameters(const Request & req, sense_voice_params & params)
{
    if (req.has_file("offset_t"))
    {
        params.offset_t_ms = std::stoi(req.get_file_value("offset_t").content);
    }
    if (req.has_file("offset_n"))
    {
        params.offset_n = std::stoi(req.get_file_value("offset_n").content);
    }
    if (req.has_file("duration"))
    {
        params.duration_ms = std::stoi(req.get_file_value("duration").content);
    }
    if (req.has_file("max_context"))
    {
        params.max_context = std::stoi(req.get_file_value("max_context").content);
    }
    if (req.has_file("max_len"))
    {
        params.max_len = std::stoi(req.get_file_value("max_len").content);
    }
    if (req.has_file("best_of"))
    {
        params.best_of = std::stoi(req.get_file_value("best_of").content);
    }
    if (req.has_file("beam_size"))
    {
        params.beam_size = std::stoi(req.get_file_value("beam_size").content);
    }
    if (req.has_file("audio_ctx"))
    {
        params.audio_ctx = std::stof(req.get_file_value("audio_ctx").content);
    }
    if (req.has_file("word_thold"))
    {
        params.word_thold = std::stof(req.get_file_value("word_thold").content);
    }
    if (req.has_file("entropy_thold"))
    {
        params.entropy_thold = std::stof(req.get_file_value("entropy_thold").content);
    }
    if (req.has_file("logprob_thold"))
    {
        params.logprob_thold = std::stof(req.get_file_value("logprob_thold").content);
    }
    if (req.has_file("debug_mode"))
    {
        params.debug_mode = parse_str_to_bool(req.get_file_value("debug_mode").content);
    }
    if (req.has_file("translate"))
    {
        params.translate = parse_str_to_bool(req.get_file_value("translate").content);
    }
    if (req.has_file("split_on_word"))
    {
        params.split_on_word = parse_str_to_bool(req.get_file_value("split_on_word").content);
    }
    if (req.has_file("no_timestamps"))
    {
        params.no_timestamps = parse_str_to_bool(req.get_file_value("no_timestamps").content);
    }
    if (req.has_file("language"))
    {
        params.language = req.get_file_value("language").content;
    }
    if (req.has_file("detect_language"))
    {
        params.detect_language = parse_str_to_bool(req.get_file_value("detect_language").content);
    }
    if (req.has_file("prompt"))
    {
        params.prompt = req.get_file_value("prompt").content;
    }
    if (req.has_file("response_format"))
    {
        params.response_format = req.get_file_value("response_format").content;
    }
    if (req.has_file("temperature"))
    {
        params.temperature = std::stof(req.get_file_value("temperature").content);
    }
    if (req.has_file("temperature_inc"))
    {
        params.temperature_inc = std::stof(req.get_file_value("temperature_inc").content);
    }
}


void sense_voice_free(struct sense_voice_context * ctx) {
    if (ctx) {
        ggml_free(ctx->model.ctx);

        ggml_backend_buffer_free(ctx->model.buffer);

        sense_voice_free_state(ctx->state);

        delete ctx->model.model->encoder;
        delete ctx->model.model;
        delete ctx;
    }
}

std::string base64_decode(const std::string& encoded) {
    const std::string base64_chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string decoded;
    int val = 0, valb = -8;
    
    for (char c : encoded) {
        if (c == '=') break;
        
        if (c > 127 || c < 0) continue;
        
        if (base64_chars.find(c) == std::string::npos) continue;
        
        val = (val << 6) + base64_chars.find(c);
        valb += 6;
        
        if (valb >= 0) {
            decoded.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    
    return decoded;
}

std::vector<float> base64_decode_to_float_array(const std::string& encoded) {
    std::string decoded = base64_decode(encoded);
    std::vector<float> pcmf32;
    pcmf32.resize(decoded.size() / sizeof(float));
    std::memcpy(pcmf32.data(), decoded.data(), decoded.size());
    return pcmf32;
}


int main(int argc, char ** argv) {
    sense_voice_params params;
    server_params sparams;

    std::mutex model_mutex;

    if (sense_voice_params_parse(argc, argv, params, sparams) == false) {
        sense_voice_print_usage(argc, argv, params, sparams);
        return 1;
    }

    if (params.language != "auto" && sense_voice_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        sense_voice_print_usage(argc, argv, params, sparams);
        exit(0);
    }

    // sense-voice init

    struct sense_voice_context_params cparams = sense_voice_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    cparams.use_itn    = params.use_itn;

    struct sense_voice_context * ctx = sense_voice_small_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize sense voice context\n");
        return 3;
    }

    ctx->language_id = sense_voice_lang_id(params.language.c_str());

    Server svr;
    svr.set_default_headers({{"Server", "whisper.cpp"},
                             {"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Headers", "content-type, authorization"}});

    std::string const default_content = R"(
    <html>
    <head>
        <title>Whisper.cpp Server</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width">
        <style>
        body {
            font-family: sans-serif;
        }
        form {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        label {
            margin-bottom: 0.5rem;
        }
        input, select {
            margin-bottom: 1rem;
        }
        button {
            margin-top: 1rem;
        }
        </style>
    </head>
    <body>
        <h1>Whisper.cpp Server</h1>

        <h2>/inference</h2>
        <pre>
    curl 127.0.0.1:)" + std::to_string(sparams.port) + R"(/inference \
    -H "Content-Type: multipart/form-data" \
    -F file="@&lt;file-path&gt;" \
    -F temperature="0.0" \
    -F temperature_inc="0.2" \
    -F response_format="json"
        </pre>

        <h2>/load</h2>
        <pre>
    curl 127.0.0.1:)" + std::to_string(sparams.port) + R"(/load \
    -H "Content-Type: multipart/form-data" \
    -F model="&lt;path-to-model-file&gt;"
        </pre>

        <div>
            <h2>Try it out</h2>
            <form action="/inference" method="POST" enctype="multipart/form-data">
                <label for="file">Choose an audio file:</label>
                <input type="file" id="file" name="file" accept="audio/*" required><br>

                <label for="temperature">Temperature:</label>
                <input type="number" id="temperature" name="temperature" value="0.0" step="0.01" placeholder="e.g., 0.0"><br>

                <label for="response_format">Response Format:</label>
                <select id="response_format" name="response_format">
                    <option value="verbose_json">Verbose JSON</option>
                    <option value="json">JSON</option>
                    <option value="text">Text</option>
                    <option value="srt">SRT</option>
                    <option value="vtt">VTT</option>
                </select><br>

                <button type="submit">Submit</button>
            </form>
        </div>
    </body>
    </html>
    )";

    // store default params so we can reset after each inference request
    sense_voice_params default_params = params;

    // this is only called if no index.html is found in the public --path
    svr.Get(sparams.request_path + "/", [&default_content](const Request &, Response &res){
        res.set_content(default_content, "text/html");
        return false;
    });

    svr.Options(sparams.request_path + sparams.inference_path, [&](const Request &, Response &){
    });

    svr.Post(sparams.request_path + sparams.inference_path, [&](const Request &req, Response &res){
        // acquire whisper model mutex lock
        std::lock_guard<std::mutex> lock(model_mutex);

        // first check user requested fields of the request
        if (!req.has_file("file"))
        {
            fprintf(stderr, "error: no 'file' field in the request\n");
            const std::string error_resp = "{\"error\":\"no 'file' field in the request\"}";
            res.set_content(error_resp, "application/json");
            return;
        }
        auto audio_file = req.get_file_value("file");

        // check non-required fields
        get_req_parameters(req, params);

        std::string filename{audio_file.filename};
        printf("Received request: %s\n", filename.c_str());

        // audio arrays
        std::vector<float> pcmf32;               // mono-channel F32 PCM
        std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

        try {
            pcmf32 = base64_decode_to_float_array(audio_file.content);
        } catch (const std::exception& e) {
            fprintf(stderr, "error: failed to decode base64 audio file: %s\n", e.what());
            const std::string error_resp = "{\"error\":\"failed to decode base64 audio file\"}";
            res.set_content(error_resp, "application/json");
            return;
        }


        printf("Successfully loaded %s\n", filename.c_str());

        // print system information
        {
            fprintf(stderr, "\n");
            fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                    params.n_threads*params.n_processors, std::thread::hardware_concurrency(), sense_voice_print_system_info());
        }

        // print some info about the processing
        {
            fprintf(stderr, "\n");
            if (params.detect_language) {
                params.language = "auto";
            }
            fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, lang = %s, task = %s, %stimestamps = %d ...\n",
                    __func__, filename.c_str(), int(pcmf32.size()), float(pcmf32.size())/SENSE_VOICE_SAMPLE_RATE,
                    params.n_threads, params.n_processors,
                    params.language.c_str(),
                    params.translate ? "translate" : "transcribe",
                    params.tinydiarize ? "tdrz = 1, " : "",
                    params.no_timestamps ? 0 : 1);

            fprintf(stderr, "\n");
        }

        sense_voice_full_params wparams = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY);
        // run inference
        {

            wparams.strategy = (params.beam_size > 1 ) ? SENSE_VOICE_SAMPLING_BEAM_SEARCH : SENSE_VOICE_SAMPLING_GREEDY;

            wparams.print_progress   = params.print_progress;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
            wparams.offset_ms        = params.offset_t_ms;
            wparams.duration_ms      = params.duration_ms;

            wparams.debug_mode       = params.debug_mode;

            wparams.greedy.best_of        = params.best_of;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.no_timestamps    = params.no_timestamps;

            // 将 pcmf32 转成 double 类型
            std::vector<double> pcmf64;
            pcmf64.resize(pcmf32.size());
            for (size_t i = 0; i < pcmf32.size(); ++i) {
                pcmf64[i] = static_cast<double>(pcmf32[i]);
            }

            if (sense_voice_full_parallel(ctx, wparams, pcmf64, pcmf64.size(), params.n_processors) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                const std::string error_resp = "{\"error\":\"failed to process audio\"}";
                res.set_content(error_resp, "application/json");
                return;
            }
        }

        std::string results = sense_voice_get_output(ctx, true, params.use_itn, false);
        // return results to user
        if (params.response_format == text_format)
        {
            res.set_content(results.c_str(), "text/html; charset=utf-8");
        } else if (params.response_format == vjson_format) {
            /* try to match openai/whisper's Python format */
            json jres = json{
                {"language", ""},
                // {"duration", float(pcmf32.size())/SENSE_VOICE_SAMPLE_RATE},
                // {"text", results},
                {"segments", json::array()}
            };
            for (int i = 0; i < 1; ++i)
            {
                json segment = json{
                    {"text", results},
                };

                if (!params.no_timestamps) {
                    segment["start"] = 0;
                    segment["end"] = float(pcmf32.size())/SENSE_VOICE_SAMPLE_RATE;
                }

                jres["segments"].push_back(segment);
            }
            res.set_content(jres.dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        }
        else
        {
            json jres = json{
                {"text", results}
            };
            res.set_content(jres.dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        }

        // reset params to their defaults
        params = default_params;
    });
    svr.Post(sparams.request_path + "/load", [&](const Request &req, Response &res){
        std::lock_guard<std::mutex> lock(model_mutex);
        if (!req.has_file("model"))
        {
            fprintf(stderr, "error: no 'model' field in the request\n");
            const std::string error_resp = "{\"error\":\"no 'model' field in the request\"}";
            res.set_content(error_resp, "application/json");
            return;
        }
        std::string model = req.get_file_value("model").content;
        if (!is_file_exist(model.c_str()))
        {
            fprintf(stderr, "error: 'model': %s not found!\n", model.c_str());
            const std::string error_resp = "{\"error\":\"model not found!\"}";
            res.set_content(error_resp, "application/json");
            return;
        }

        // clean up
        sense_voice_free(ctx);

        // whisper init
        ctx = sense_voice_small_init_from_file_with_params(model.c_str(), cparams);

        // TODO perhaps load prior model here instead of exit
        if (ctx == nullptr) {
            fprintf(stderr, "error: model init  failed, no model loaded must exit\n");
            exit(1);
        }

        const std::string success = "Load was successful!";
        res.set_content(success, "application/text");

        // check if the model is in the file system
    });

    svr.set_exception_handler([](const Request &, Response &res, std::exception_ptr ep) {
        const char fmt[] = "500 Internal Server Error\n%s";
        char buf[BUFSIZ];
        try {
            std::rethrow_exception(std::move(ep));
        } catch (std::exception &e) {
            snprintf(buf, sizeof(buf), fmt, e.what());
        } catch (...) {
            snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
        }
        res.set_content(buf, "text/plain");
        res.status = 500;
    });

    svr.set_error_handler([](const Request &req, Response &res) {
        if (res.status == 400) {
            res.set_content("Invalid request", "text/plain");
        } else if (res.status != 500) {
            res.set_content("File Not Found (" + req.path + ")", "text/plain");
            res.status = 404;
        }
    });

    // set timeouts and change hostname and port
    svr.set_read_timeout(sparams.read_timeout);
    svr.set_write_timeout(sparams.write_timeout);

    int port = sparams.port;
    if (sparams.port == 0) {
        port = svr.bind_to_any_port(sparams.hostname);
    } else {
        if (!svr.bind_to_port(sparams.hostname, sparams.port))
        {
            fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n",
                sparams.hostname.c_str(), sparams.port);
            return 1;
        }
    }

    // Set the base directory for serving static files
    svr.set_base_dir(sparams.public_path);

    // to make it ctrl+clickable:
    fprintf(stderr, "\nwhisper server listening at http://%s:%d\n\n", sparams.hostname.c_str(), port);

    if (!svr.listen_after_bind())
    {
        return 1;
    }

    sense_voice_free(ctx);

    return 0;
}
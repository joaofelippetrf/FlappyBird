/**
 * vision.cpp
 *
 * Camada de percepção visual para o agente Flappy Bird embarcado.
 * Pipeline: captura → grayscale → binarização → Canny → extração de estado
 *
 * Dependências: OpenCV 4.x
 * Compilar: g++ -std=c++17 vision.cpp -o vision `pkg-config --cflags --libs opencv4`
 *
 * Uso:
 *   ./vision [camera_index] [delay_ms] [truth_path]
 *
 *   camera_index  índice da câmera (default 0)
 *   delay_ms      atraso da câmera em relação ao jogo (default 0)
 *   truth_path    JSONL com ground truth (default /tmp/flappy_truth.jsonl)
 *
 * Para validação contra o jogo: rode FlappyBird (que escreve em /tmp/flappy_truth.jsonl)
 * e em paralelo ./vision com a câmera apontada pra tela.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <deque>
#include <optional>
#include <chrono>
#include <string>

// Cérebro do agente (rede neural treinada) + envio UDP do pulo.
#include "Brain.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// ─────────────────────────────────────────────
// PARÂMETROS DE CONFIGURAÇÃO
// ─────────────────────────────────────────────

// Índice padrão da câmera (sobrescrito pelo 1º argumento de linha de comando)
// Em Macs com Continuity Camera / câmeras virtuais, a numeração pode variar.
static const int DEFAULT_CAMERA_INDEX = 0;

// Resolução de captura
static const int FRAME_W = 640;
static const int FRAME_H = 480;

// Binarização (Equação 2 do documento)
static const int THRESH_VALUE = 100;   // limiar 'thresh'
static const int THRESH_MAX   = 255;   // 'maxval'

// Canny (Equação 3 do documento)
static const int CANNY_LOW  = 50;
static const int CANNY_HIGH = 150;

// Área mínima de contorno para filtrar ruído
static const double MIN_CONTOUR_AREA = 100.0;

// ─────────────────────────────────────────────
// ESTRUTURA DE ESTADO DO JOGO
// ─────────────────────────────────────────────

struct GameState {
    // Posição normalizada do pássaro [0.0, 1.0]
    float bird_x;
    float bird_y;

    // Posição e abertura do próximo cano [0.0, 1.0]
    float pipe_x;          // borda esquerda do cano
    float pipe_gap_top;    // topo da abertura
    float pipe_gap_bottom; // base da abertura

    // Distâncias relativas (entrada X da rede neural)
    float dist_x;          // distância horizontal até o cano
    float dist_y_top;      // distância vertical até o topo do gap
    float dist_y_bottom;   // distância vertical até o fundo do gap

    float camera_dx;       // deriva do bird_x observado vs. âncora (estimativa de pan da câmera)

    bool valid;            // frame processado com sucesso
};

// Bounding box enriquecido com métricas de forma — circularidade descarta
// retângulos verticais (tampa/lip de cano) que passam no aspect-ratio "quadradinho".
struct DetectedObject {
    cv::Rect box;
    double area;
    double perimeter;
    double circularity;    // 4π·area / perim²  ∈ (0, 1]  (1 = círculo perfeito)
};

// Âncora temporal do bird_x. O pássaro é fixo em x no jogo, então uma EMA
// das observações válidas converge pra posição real (e qualquer drift = câmera mexendo).
struct BirdTracker {
    bool  has_anchor       = false;
    float anchor_x         = 0.0f;   // EMA atualizada a cada frame válido
    float lock_x           = 0.0f;   // âncora no momento do lock — referência fixa
    int   frames_observed  = 0;      // observações coerentes acumuladas
    int   lock_after       = 15;     // observações até travar a âncora
    float ema_alpha        = 0.05f;  // suavização lenta (bird não anda em x)
    float search_window    = 0.10f;  // ±10% da largura ao redor da âncora
    float min_circularity  = 0.55f;  // descarta retângulos (cano/lip)
};

// ─────────────────────────────────────────────
// VALIDAÇÃO — ground truth vindo de main.cpp
// ─────────────────────────────────────────────

struct GroundTruth {
    double t;                 // segundos desde epoch
    float bird_x, bird_y, bird_r;
    float pipe_x, gap_top, gap_bottom;
    bool  alive;
};

static double wallSeconds() {
    auto d = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration<double>(d).count();
}

// Parser minimalista para nossa JSONL fixa (chaves planas, ordem livre, números).
static std::optional<GroundTruth> parseTruthLine(const std::string& line) {
    auto get = [&](const char* key) -> std::optional<double> {
        std::string pat = std::string("\"") + key + "\":";
        size_t p = line.find(pat);
        if (p == std::string::npos) return std::nullopt;
        p += pat.size();
        size_t end = line.find_first_of(",}", p);
        if (end == std::string::npos) return std::nullopt;
        try { return std::stod(line.substr(p, end - p)); }
        catch (...) { return std::nullopt; }
    };
    auto t  = get("t");           auto bx = get("bird_x");
    auto by = get("bird_y");      auto br = get("bird_r");
    auto px = get("pipe_x");      auto gt = get("gap_top");
    auto gb = get("gap_bottom");  auto al = get("alive");
    if (!t || !bx || !by || !px || !gt || !gb) return std::nullopt;
    GroundTruth g{};
    g.t = *t;
    g.bird_x = (float)*bx;  g.bird_y = (float)*by;
    g.bird_r = br ? (float)*br : 0.02f;
    g.pipe_x = (float)*px;
    g.gap_top = (float)*gt;  g.gap_bottom = (float)*gb;
    g.alive = al ? (*al > 0.5) : true;
    return g;
}

// Tail incremental do JSONL. Mantém um buffer dos últimos N estados.
class TruthTailer {
    std::ifstream in;
    std::string path;
    std::deque<GroundTruth> buf;
    static constexpr size_t MAX_BUF = 600;  // ~10s a 60Hz
public:
    explicit TruthTailer(std::string p) : path(std::move(p)) { reopen(); }

    void reopen() { in.close(); in.clear(); in.open(path); }

    void poll() {
        if (!in.is_open()) { reopen(); if (!in.is_open()) return; }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (auto g = parseTruthLine(line)) {
                buf.push_back(*g);
                if (buf.size() > MAX_BUF) buf.pop_front();
            }
        }
        in.clear();  // limpa EOF pra continuar lendo no próximo poll
    }

    // Retorna a entrada cujo timestamp está mais próximo de (now - delay).
    std::optional<GroundTruth> closestTo(double target) const {
        if (buf.empty()) return std::nullopt;
        const GroundTruth* best = &buf.front();
        double bestErr = std::abs(best->t - target);
        for (const auto& g : buf) {
            double err = std::abs(g.t - target);
            if (err < bestErr) { best = &g; bestErr = err; }
        }
        return *best;
    }

    size_t size() const { return buf.size(); }
};

// Acumulador online de erro absoluto médio (MAE).
struct MetricAccum {
    double sum = 0.0;
    size_t n = 0;
    void add(double e) { sum += std::abs(e); ++n; }
    double mae() const { return n ? sum / n : 0.0; }
};

// ─────────────────────────────────────────────
// FUNÇÕES DO PIPELINE DE VISÃO
// ─────────────────────────────────────────────

/**
 * Passo 1 — Conversão para escala de cinza (Equação 1)
 * I(x,y) = 0.299·R + 0.587·G + 0.114·B
 * (OpenCV já aplica esses coeficientes em cvtColor)
 */
cv::Mat toGrayscale(const cv::Mat& frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

/**
 * Passo 2 — Binarização (Equação 2)
 * dst(x,y) = maxval se src(x,y) > thresh, senão 0
 */
cv::Mat binarize(const cv::Mat& gray) {
    cv::Mat binary;
    cv::threshold(gray, binary, THRESH_VALUE, THRESH_MAX, cv::THRESH_BINARY);
    return binary;
}

/**
 * Passo 3 — Detecção de bordas Canny (Equação 3)
 * Calcula gradientes Gx e Gy via filtros Sobel,
 * G = sqrt(Gx² + Gy²), θ = atan(Gy/Gx)
 */
cv::Mat detectEdges(const cv::Mat& binary) {
    cv::Mat edges;
    cv::Canny(binary, edges, CANNY_LOW, CANNY_HIGH);
    return edges;
}

/**
 * Passo 4 — Extração de contornos e coordenadas (Seção 2.1.4)
 * Retorna os bounding boxes dos objetos detectados.
 */
std::vector<DetectedObject> extractContours(const cv::Mat& edges) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<DetectedObject> objects;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area <= MIN_CONTOUR_AREA) continue;
        double perim = cv::arcLength(c, /*closed=*/true);
        double circ  = perim > 1e-6 ? (4.0 * CV_PI * area) / (perim * perim) : 0.0;
        objects.push_back({cv::boundingRect(c), area, perim, circ});
    }
    return objects;
}

/**
 * Interpreta os bounding boxes para separar pássaro e canos.
 *
 * Heurística:
 *  - Pássaro: objeto mais à esquerda com aspect ratio ≈ quadrado
 *  - Canos:   objetos altos (altura > largura * 2), agrupados em pares (top/bottom)
 *
 * Normaliza todas as coordenadas para [0.0, 1.0].
 */
GameState interpretScene(const std::vector<DetectedObject>& objects,
                         int W, int H, BirdTracker& tracker) {
    GameState state{};
    state.valid = false;

    if (objects.empty()) return state;

    std::vector<cv::Rect> pipe_boxes;
    std::vector<const DetectedObject*> bird_candidates;

    for (const auto& o : objects) {
        const cv::Rect& r = o.box;
        float aspect = (float)r.width / (float)r.height;
        bool is_tall = r.height > r.width * 1.5f;

        if (is_tall) {
            // Provavelmente é um cano
            pipe_boxes.push_back(r);
        } else if (aspect > 0.4f && aspect < 2.5f &&
                   o.circularity >= tracker.min_circularity) {
            // Candidato a pássaro: quadradinho E circular o bastante
            // (descarta tampas/lips de cano, que têm circularidade baixa).
            bird_candidates.push_back(&o);
        }
    }

    if (bird_candidates.empty() || pipe_boxes.empty()) return state;

    // Tamanho do menor cano serve de teto pro tamanho do pássaro:
    // o pássaro é sempre menor que um cano no quadro.
    int min_pipe_h = std::numeric_limits<int>::max();
    for (const auto& p : pipe_boxes) min_pipe_h = std::min(min_pipe_h, p.height);

    const DetectedObject* picked = nullptr;
    if (tracker.has_anchor) {
        // Âncora travada: o melhor candidato é o mais perto do anchor_x,
        // dentro da janela de busca. Resolve a confusão com a base do cano.
        float best_dx = tracker.search_window;
        for (const auto* c : bird_candidates) {
            if (c->box.height >= min_pipe_h) continue;        // grande demais
            float cx = (c->box.x + c->box.width * 0.5f) / W;
            float dx = std::abs(cx - tracker.anchor_x);
            if (dx < best_dx) { best_dx = dx; picked = c; }
        }
    }
    if (!picked) {
        // Sem âncora (ou nenhum candidato na janela): pega o mais circular
        // entre os de tamanho plausível.
        double best_circ = -1.0;
        for (const auto* c : bird_candidates) {
            if (c->box.height >= min_pipe_h) continue;
            if (c->circularity > best_circ) { best_circ = c->circularity; picked = c; }
        }
    }
    if (!picked) return state;

    cv::Rect bird_box = picked->box;

    // Centro do pássaro (normalizado)
    state.bird_x = (bird_box.x + bird_box.width  / 2.0f) / W;
    state.bird_y = (bird_box.y + bird_box.height / 2.0f) / H;

    // Encontra o par de canos mais próximo à direita do pássaro
    // Separa canos superiores (y pequeno) e inferiores (y grande)
    cv::Rect top_pipe{-1,-1,0,0}, bot_pipe{-1,-1,0,0};
    int closest_x = W + 1;

    for (const auto& p : pipe_boxes) {
        if (p.x > bird_box.x && p.x < closest_x) {
            closest_x = p.x;
            // Decide se é cano de cima ou de baixo pelo centro vertical
            float cy = p.y + p.height / 2.0f;
            if (cy < H / 2.0f)
                top_pipe = p;
            else
                bot_pipe = p;
        }
    }

    // Requer pelo menos um dos canos detectado
    if (top_pipe.x == -1 && bot_pipe.x == -1) return state;

    // Posição X do cano (borda esquerda, normalizada)
    int pipe_ref_x = (top_pipe.x != -1) ? top_pipe.x : bot_pipe.x;
    state.pipe_x = (float)pipe_ref_x / W;

    // Bordas do gap (abertura entre os canos)
    state.pipe_gap_top    = (top_pipe.x != -1)
                            ? (float)(top_pipe.y + top_pipe.height) / H
                            : 0.0f;
    state.pipe_gap_bottom = (bot_pipe.x != -1)
                            ? (float)bot_pipe.y / H
                            : 1.0f;

    // Distâncias relativas (vetor X da rede neural)
    state.dist_x        = state.pipe_x      - state.bird_x;
    state.dist_y_top    = state.bird_y      - state.pipe_gap_top;
    state.dist_y_bottom = state.pipe_gap_bottom - state.bird_y;

    // Atualiza âncora EMA. Antes do lock, acumula observações; depois do lock,
    // só absorve observações dentro da janela (gate contra outliers).
    if (!tracker.has_anchor) {
        tracker.anchor_x = (tracker.frames_observed == 0)
            ? state.bird_x
            : tracker.anchor_x + tracker.ema_alpha * (state.bird_x - tracker.anchor_x);
        if (++tracker.frames_observed >= tracker.lock_after) {
            tracker.has_anchor = true;
            tracker.lock_x = tracker.anchor_x;
        }
        state.camera_dx = 0.0f;
    } else {
        float dx_obs = state.bird_x - tracker.anchor_x;
        if (std::abs(dx_obs) < tracker.search_window) {
            tracker.anchor_x += tracker.ema_alpha * dx_obs;
        }
        // Diferença entre âncora atual e onde ela travou = pan da câmera.
        state.camera_dx = tracker.anchor_x - tracker.lock_x;
    }

    state.valid = true;
    return state;
}

// ─────────────────────────────────────────────
// VISUALIZAÇÃO (debug)
// ─────────────────────────────────────────────

void drawDebug(cv::Mat& frame, const GameState& s,
               const std::vector<DetectedObject>& objects,
               const BirdTracker& tracker) {
    int W = frame.cols, H = frame.rows;

    // Classificação visual usando a mesma heurística de interpretScene:
    //   verde     → cano  (altura > 1.5 × largura)
    //   vermelho  → candidato a pássaro (quadradinho E circular)
    //   cinza     → ruído (descartado pela lógica)
    for (const auto& o : objects) {
        const cv::Rect& r = o.box;
        float aspect = (float)r.width / (float)r.height;
        bool is_tall = r.height > r.width * 1.5f;
        bool is_squarish = (aspect > 0.4f && aspect < 2.5f);
        bool is_round    = o.circularity >= tracker.min_circularity;

        cv::Scalar color;
        int thickness = 2;
        if (is_tall) {
            color = cv::Scalar(0, 255, 0);        // verde — cano
        } else if (is_squarish && is_round) {
            color = cv::Scalar(0, 0, 255);        // vermelho — pássaro
        } else {
            color = cv::Scalar(120, 120, 120);    // cinza — ruído
            thickness = 1;
        }
        cv::rectangle(frame, r, color, thickness);
    }

    // Linha vertical da âncora (magenta) — referência do bird_x travado.
    if (tracker.has_anchor) {
        int ax = (int)(tracker.anchor_x * W);
        cv::line(frame, {ax, 0}, {ax, H}, cv::Scalar(255, 0, 255), 1);
    }

    if (!s.valid) {
        cv::putText(frame, "ESTADO INVALIDO", {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,0,255}, 2);
        return;
    }

    // Pássaro
    int bx = (int)(s.bird_x * W), by = (int)(s.bird_y * H);
    cv::circle(frame, {bx, by}, 8, cv::Scalar(0, 0, 255), -1);

    // Linhas do gap
    int px = (int)(s.pipe_x * W);
    cv::line(frame, {px, (int)(s.pipe_gap_top * H)},
                    {W,  (int)(s.pipe_gap_top * H)}, {255,255,0}, 1);
    cv::line(frame, {px, (int)(s.pipe_gap_bottom * H)},
                    {W,  (int)(s.pipe_gap_bottom * H)}, {255,255,0}, 1);

    // HUD com estado
    auto txt = [&](const std::string& s, int line) {
        cv::putText(frame, s, {10, 20 + line * 20},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, {255,255,255}, 1);
    };
    txt("bird  x=" + std::to_string(s.bird_x).substr(0,4) +
              " y=" + std::to_string(s.bird_y).substr(0,4), 0);
    txt("pipe  x=" + std::to_string(s.pipe_x).substr(0,4), 1);
    txt("gap   top="    + std::to_string(s.pipe_gap_top).substr(0,4) +
              " bot="   + std::to_string(s.pipe_gap_bottom).substr(0,4), 2);
    txt("dist  dx=" + std::to_string(s.dist_x).substr(0,5) +
              " dy_top=" + std::to_string(s.dist_y_top).substr(0,5), 3);
    txt(std::string("anchor ") + (tracker.has_anchor ? "LOCK" : "...") +
        " x=" + std::to_string(tracker.anchor_x).substr(0,4) +
        " cam_dx=" + std::to_string(s.camera_dx).substr(0,5), 4);
}

// ─────────────────────────────────────────────
// AGENTE EMBARCADO — carrega o cérebro e envia o pulo por UDP
// ─────────────────────────────────────────────

// Carrega um Brain do arquivo texto salvo por main.cpp (mesma ordem/formato).
static bool loadBrain(Brain& out, const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    int ni, nh; f >> ni >> nh;
    if (!f || ni <= 0 || nh <= 0) return false;
    std::vector<double> w_h(ni * nh), b_h(nh), w_o(nh);
    double b_o = 0.0;
    for (auto& x : w_h) f >> x;
    for (auto& x : b_h) f >> x;
    for (auto& x : w_o) f >> x;
    f >> b_o;
    if (!f) return false;
    out = Brain(w_h, b_h, w_o, b_o);
    out.numInput  = ni;
    out.numHidden = nh;
    return true;
}

// Socket UDP destino (PC rodando o jogo em modo serve).
struct JumpSender {
    int fd = -1;
    sockaddr_in dst{};
    bool open(const std::string& ip, unsigned short port) {
        fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (fd < 0) return false;
        dst.sin_family = AF_INET;
        dst.sin_port = htons(port);
        if (inet_pton(AF_INET, ip.c_str(), &dst.sin_addr) != 1) { ::close(fd); fd = -1; return false; }
        return true;
    }
    void sendJump() {
        if (fd < 0) return;
        const char msg = '1';
        sendto(fd, &msg, 1, 0, reinterpret_cast<sockaddr*>(&dst), sizeof(dst));
    }
    ~JumpSender() { if (fd >= 0) ::close(fd); }
};

// Constantes do jogo (espelham Agent.cpp / main.cpp) para converter o estado
// normalizado da visão nas 4 entradas exatas que o cérebro espera.
static const double GAME_SCREEN_H = 1080.0;
static const double GAME_SCREEN_W = 500.0;   // normalizador de distX no Agent.cpp
static const double GAME_MAX_VEL  = 600.0;
static const double VEL_SCALE     = GAME_SCREEN_H / GAME_MAX_VEL;       // 1.8
static const double DISTX_SCALE   = 1920.0 / GAME_SCREEN_W;            // 3.84
static const double AGENT_COOLDOWN = 0.1;     // mesmo cooldown do Agent.hpp

// ─────────────────────────────────────────────
// LOOP PRINCIPAL
// ─────────────────────────────────────────────

int main(int argc, char** argv) {
    int camera_index   = (argc > 1) ? std::atoi(argv[1]) : DEFAULT_CAMERA_INDEX;
    double delay_ms    = (argc > 2) ? std::atof(argv[2]) : 0.0;   // câmera atrasa N ms vs jogo
    std::string truth_path = (argc > 3) ? argv[3] : "/tmp/flappy_truth.jsonl";

    // Modo agente: ./vision <cam> <delay> <truth> --agent <pc_ip> <porta> <brain_path>
    bool agent_mode = false;
    std::string pc_ip = "127.0.0.1";
    unsigned short pc_port = 5005;
    std::string brain_path = "best_brain.txt";
    for (int i = 4; i < argc; i++) {
        if (std::string(argv[i]) == "--agent") {
            agent_mode = true;
            if (i + 1 < argc) pc_ip     = argv[i + 1];
            if (i + 2 < argc) pc_port   = (unsigned short)std::atoi(argv[i + 2]);
            if (i + 3 < argc) brain_path = argv[i + 3];
            break;
        }
    }

    Brain brain;
    JumpSender sender;
    if (agent_mode) {
        if (!loadBrain(brain, brain_path)) {
            std::cerr << "[ERRO] Nao foi possivel carregar o cerebro de '" << brain_path << "'\n";
            return -1;
        }
        if (!sender.open(pc_ip, pc_port)) {
            std::cerr << "[ERRO] Nao foi possivel abrir socket UDP para " << pc_ip << ":" << pc_port << "\n";
            return -1;
        }
        std::cout << "[AGENT] cerebro=" << brain_path
                  << " -> envia pulos para " << pc_ip << ":" << pc_port << "\n";
    }

    // Estado para estimar a velocidade vertical entre frames.
    bool  have_prev_bird = false;
    float prev_bird_y = 0.0f;
    double prev_t = 0.0;
    double vel_ema = 0.0;            // velocidade suavizada (já na escala do cérebro)
    double agent_cooldown = 0.0;     // trava entre pulos (igual ao Agent)

    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        std::cerr << "[ERRO] Nao foi possivel abrir a camera (index "
                  << camera_index << ")\n";
        return -1;
    }
    std::cout << "[INFO] Usando camera index " << camera_index
              << " | delay=" << delay_ms << "ms"
              << " | truth=" << truth_path << "\n";

    cap.set(cv::CAP_PROP_FRAME_WIDTH,  FRAME_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_H);

    TruthTailer truth(truth_path);
    std::ofstream csv("/tmp/flappy_validation.csv", std::ios::out | std::ios::trunc);
    if (csv) {
        csv << "t_cam,t_gt,bird_x_est,bird_y_est,bird_x_gt,bird_y_gt,"
               "pipe_x_est,pipe_x_gt,gap_top_est,gap_top_gt,"
               "gap_bot_est,gap_bot_gt,err_bird_x,err_bird_y,err_pipe_x,valid\n";
    }

    MetricAccum mae_bx, mae_by, mae_px;
    size_t frames_total = 0, frames_valid = 0;

    std::cout << "[INFO] Camera inicializada. Pressione 'q' para sair.\n";

    cv::Mat frame, gray, binary, edges;
    BirdTracker tracker;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "[AVISO] Frame vazio — verificar camera.\n";
            continue;
        }

        // ── Pipeline de visão ──────────────────
        gray   = toGrayscale(frame);
        binary = binarize(gray);
        edges  = detectEdges(binary);

        auto objects = extractContours(edges);
        GameState state = interpretScene(objects, frame.cols, frame.rows, tracker);
        // ──────────────────────────────────────

        // ── Agente: estado → 4 entradas do cérebro → decide → UDP ──
        if (agent_mode) {
            double t_now = wallSeconds();
            if (state.valid) {
                // Velocidade vertical estimada por diferença de posição entre frames.
                if (have_prev_bird) {
                    double dt_s = t_now - prev_t;
                    if (dt_s > 1e-4) {
                        double rate = (state.bird_y - prev_bird_y) / dt_s;   // d(bird_y_norm)/dt
                        double vel  = VEL_SCALE * rate;                      // escala do cérebro
                        vel_ema = 0.4 * vel + 0.6 * vel_ema;                 // suaviza ruído
                    }
                    agent_cooldown -= dt_s;
                }
                prev_bird_y = state.bird_y;
                prev_t = t_now;
                have_prev_bird = true;

                // Conversão exata p/ as unidades de Agent.cpp:
                //   nw = bird_y ; vel = vel_ema ; distX = 3.84*(pipe-bird) ; distY = gap_top-bird
                // (pipe_x é a borda esquerda do cano; calibrar o offset vs. o centro se preciso)
                std::vector<double> input = {
                    (double)state.bird_y,
                    vel_ema,
                    DISTX_SCALE * (state.pipe_x - state.bird_x),
                    (double)(state.pipe_gap_top - state.bird_y)
                };

                if (agent_cooldown <= 0.0 && brain.jump(input)) {
                    sender.sendJump();
                    agent_cooldown = AGENT_COOLDOWN;
                }
            } else {
                have_prev_bird = false;   // perdeu o pássaro: zera o histórico de velocidade
            }
        }

        // ── Ground truth (validação) ───────────
        truth.poll();
        double t_cam = wallSeconds();
        auto gt = truth.closestTo(t_cam - delay_ms / 1000.0);

        ++frames_total;
        if (state.valid) ++frames_valid;

        // Janelas de debug
        drawDebug(frame, state, objects, tracker);

        if (gt) {
            // Overlay verde = verdade absoluta do jogo
            int W = frame.cols, H = frame.rows;
            int bx_gt = (int)(gt->bird_x * W);
            int by_gt = (int)(gt->bird_y * H);
            int rad   = std::max(5, (int)(gt->bird_r * H));
            cv::circle(frame, {bx_gt, by_gt}, rad, cv::Scalar(0, 255, 0), 2);
            int px_gt = (int)(gt->pipe_x * W);
            int gy_t  = (int)(gt->gap_top    * H);
            int gy_b  = (int)(gt->gap_bottom * H);
            cv::line(frame, {px_gt, gy_t}, {W, gy_t}, cv::Scalar(0,255,0), 1);
            cv::line(frame, {px_gt, gy_b}, {W, gy_b}, cv::Scalar(0,255,0), 1);
            cv::line(frame, {px_gt, 0}, {px_gt, H},   cv::Scalar(0,255,0), 1);

            // Linha do erro: estimativa → verdade
            if (state.valid) {
                cv::line(frame,
                         {(int)(state.bird_x * W), (int)(state.bird_y * H)},
                         {bx_gt, by_gt},
                         cv::Scalar(0, 255, 255), 2);
                mae_bx.add(state.bird_x - gt->bird_x);
                mae_by.add(state.bird_y - gt->bird_y);
                mae_px.add(state.pipe_x - gt->pipe_x);
            }

            // Log CSV
            if (csv) {
                csv << std::fixed << std::setprecision(5)
                    << t_cam << "," << gt->t << ","
                    << (state.valid ? state.bird_x : NAN) << ","
                    << (state.valid ? state.bird_y : NAN) << ","
                    << gt->bird_x << "," << gt->bird_y << ","
                    << (state.valid ? state.pipe_x : NAN) << ","
                    << gt->pipe_x << ","
                    << (state.valid ? state.pipe_gap_top : NAN) << ","
                    << gt->gap_top << ","
                    << (state.valid ? state.pipe_gap_bottom : NAN) << ","
                    << gt->gap_bottom << ","
                    << (state.valid ? (state.bird_x - gt->bird_x) : NAN) << ","
                    << (state.valid ? (state.bird_y - gt->bird_y) : NAN) << ","
                    << (state.valid ? (state.pipe_x - gt->pipe_x) : NAN) << ","
                    << (state.valid ? 1 : 0) << "\n";
            }
        }

        // HUD de validação no canto inferior
        {
            auto put = [&](const std::string& s, int row) {
                cv::putText(frame, s, {10, frame.rows - 10 - row * 18},
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 255), 1);
            };
            char line[160];
            std::snprintf(line, sizeof(line),
                "valid=%zu/%zu (%.0f%%)  gt_buf=%zu",
                frames_valid, frames_total,
                frames_total ? 100.0 * frames_valid / frames_total : 0.0,
                truth.size());
            put(line, 2);
            std::snprintf(line, sizeof(line),
                "MAE bird_x=%.3f bird_y=%.3f pipe_x=%.3f  (delay=%.0fms)",
                mae_bx.mae(), mae_by.mae(), mae_px.mae(), delay_ms);
            put(line, 1);
            put(gt ? "GT=on (verde)" : "GT=off (rode ./FlappyBird)", 0);
        }

        cv::imshow("Flappy - Camera",  frame);
        cv::imshow("Flappy - Edges",   edges);
        cv::imshow("Flappy - Binary",  binary);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "[INFO] CSV salvo em /tmp/flappy_validation.csv\n";
    return 0;
}
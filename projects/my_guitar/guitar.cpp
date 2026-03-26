#include "RtAudio.h"
#include "Stk.h"
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <stdexcept>
#include <termios.h>
#include <unistd.h>

using namespace std;
using namespace stk;

constexpr StkFloat sample_rate = 48'000.0;
constexpr size_t n_channels = 2;

namespace {

class BiQuad {
    StkFloat a1;
    StkFloat a2;
    StkFloat b0;
    StkFloat b1;
    StkFloat b2;

    StkFloat s1 = 0.0;
    StkFloat s2 = 0.0;

    BiQuad(StkFloat a1, StkFloat a2, StkFloat b0, StkFloat b1, StkFloat b2)
        : a1(a1), a2(a2), b0(b0), b1(b1), b2(b2) {}

  public:
    static auto make_allpass(StkFloat f0, StkFloat Q) -> BiQuad {
        const StkFloat w0 = 2.0 * numbers::pi * f0 / sample_rate;
        const StkFloat alpha = sin(w0) / (2.0 * Q);
        const StkFloat a0 = 1.0 + alpha;
        const StkFloat a1 = -2.0 * cos(w0);
        const StkFloat a2 = 1.0 - alpha;
        const StkFloat b0 = a2;
        const StkFloat b1 = a1;
        const StkFloat b2 = a0;
        return {a1 / a0, a2 / a0, b0 / a0, b1 / a0, b2 / a0};
    }

    static auto make_lowpass(StkFloat f0, StkFloat Q) -> BiQuad {
        const StkFloat w0 = 2.0 * numbers::pi * f0 / sample_rate;
        const StkFloat alpha = sin(w0) / (2.0 * Q);
        const StkFloat a0 = 1.0 + alpha;
        const StkFloat a1 = -2.0 * cos(w0);
        const StkFloat a2 = 1.0 - alpha;
        const StkFloat b0 = 0.5 * (1.0 - cos(w0));
        const StkFloat b1 = 1.0 - cos(w0);
        const StkFloat b2 = 0.5 * (1.0 - cos(w0));
        return {a1 / a0, a2 / a0, b0 / a0, b1 / a0, b2 / a0};
    }

    static auto make_bandpass(StkFloat f0, StkFloat Q) -> BiQuad {
        const StkFloat w0 = 2.0 * numbers::pi * f0 / sample_rate;
        const StkFloat alpha = sin(w0) / (2.0 * Q);
        const StkFloat a0 = 1.0 + alpha;
        const StkFloat a1 = -2.0 * cos(w0);
        const StkFloat a2 = 1.0 - alpha;
        const StkFloat b0 = alpha;
        const StkFloat b1 = 0.0;
        const StkFloat b2 = -alpha;
        return {a1 / a0, a2 / a0, b0 / a0, b1 / a0, b2 / a0};
    }

    static auto make_highpass(StkFloat f0, StkFloat Q) -> BiQuad {
        const StkFloat w0 = 2.0 * numbers::pi * f0 / sample_rate;
        const StkFloat alpha = sin(w0) / (2.0 * Q);
        const StkFloat a0 = 1.0 + alpha;
        const StkFloat a1 = -2.0 * cos(w0);
        const StkFloat a2 = 1.0 - alpha;
        const StkFloat b0 = 0.5 * (1.0 + cos(w0));
        const StkFloat b1 = -(1.0 + cos(w0));
        const StkFloat b2 = 0.5 * (1.0 + cos(w0));
        return {a1 / a0, a2 / a0, b0 / a0, b1 / a0, b2 / a0};
    }

    auto tick(StkFloat x) -> StkFloat {
        const StkFloat y = (b0 * x) + s1;
        s1 = (b1 * x) - (a1 * y) + s2;
        s2 = (b2 * x) - (a2 * y);
        return y;
    }

    [[nodiscard]] auto phase_delay(StkFloat f) const -> StkFloat {
        const StkFloat w = 2.0 * numbers::pi * f / sample_rate;
        const StkFloat re_b = b0 + (b1 * cos(w)) + (b2 * cos(2.0 * w));
        const StkFloat im_b = -((b1 * sin(w)) + (b2 * sin(2.0 * w)));
        const StkFloat re_a = 1.0 + (a1 * cos(w)) + (a2 * cos(2.0 * w));
        const StkFloat im_a = -((a1 * sin(w)) + (a2 * sin(2.0 * w)));
        const StkFloat phase = atan2(im_b, re_b) - atan2(im_a, re_a);
        return -phase / w;
    }

    [[nodiscard]] auto magnitude_at(StkFloat f) const -> StkFloat {
        const StkFloat w = 2.0 * numbers::pi * f / sample_rate;
        const StkFloat re_b = b0 + (b1 * cos(w)) + (b2 * cos(2.0 * w));
        const StkFloat im_b = -((b1 * sin(w)) + (b2 * sin(2.0 * w)));
        const StkFloat re_a = 1.0 + (a1 * cos(w)) + (a2 * cos(2.0 * w));
        const StkFloat im_a = -((a1 * sin(w)) + (a2 * sin(2.0 * w)));
        return sqrt((re_b * re_b + im_b * im_b) / (re_a * re_a + im_a * im_a));
    }
};

class String {
    vector<StkFloat> p2b;
    vector<StkFloat> p2n;
    vector<StkFloat> b2p;
    vector<StkFloat> n2p;
    size_t p2b_idx = 0;
    size_t p2n_idx = 0;
    size_t b2p_idx = 0;
    size_t n2p_idx = 0;
    std::array<BiQuad, 4> p2b_dispersion = {
        BiQuad::make_allpass(3'500.0, 0.8),
        BiQuad::make_allpass(7'000.0, 0.8),
        BiQuad::make_allpass(11'000.0, 0.8),
        BiQuad::make_allpass(15'000.0, 0.8),
    };
    std::array<BiQuad, 4> p2n_dispersion = {
        BiQuad::make_allpass(3'500.0, 0.8),
        BiQuad::make_allpass(7'000.0, 0.8),
        BiQuad::make_allpass(11'000.0, 0.8),
        BiQuad::make_allpass(15'000.0, 0.8),
    };
    BiQuad b2p_filter_lo = BiQuad::make_lowpass(12'000.0, 0.7);
    BiQuad n2p_filter_lo = BiQuad::make_lowpass(12'000.0, 0.7);
    BiQuad b2p_filter_hi = BiQuad::make_lowpass(18'000.0, 0.7);
    BiQuad n2p_filter_hi = BiQuad::make_lowpass(18'000.0, 0.7);

    StkFloat attenuation = 1.0;
    StkFloat tension_coeff;
    StkFloat dc_x1 = 0.0;
    StkFloat dc_y1 = 0.0;

    const size_t oversample;

  public:
    explicit String(StkFloat f0, StkFloat t60, StkFloat pick_position,
                    StkFloat tension_coeff = 2.0, size_t oversample = 4)
        : tension_coeff(tension_coeff), oversample(oversample) {
        if (t60 <= 0.0) throw invalid_argument("Invalid t60!");

        if (pick_position <= 0.0 || pick_position >= 1.0)
            throw invalid_argument("Invalid pick position!");

        const auto base_del =
            static_cast<StkFloat>(oversample) * sample_rate / (2.0 * f0);

        auto p2b_del = (1.0 - pick_position) * base_del;
        for (const auto &f : p2b_dispersion) p2b_del -= f.phase_delay(f0);

        auto p2n_del = pick_position * base_del;
        for (const auto &f : p2n_dispersion) p2n_del -= f.phase_delay(f0);

        auto b2p_del = (1.0 - pick_position) * base_del;
        b2p_del -= b2p_filter_lo.phase_delay(f0);
        b2p_del -= b2p_filter_hi.phase_delay(f0);

        auto n2p_del = pick_position * base_del;
        n2p_del -= n2p_filter_lo.phase_delay(f0);
        n2p_del -= n2p_filter_hi.phase_delay(f0);

        if (p2b_del < 4.0 || b2p_del < 4.0 || p2n_del < 4.0 || n2p_del < 4.0)
            throw invalid_argument("Frequency too high!");

        p2b.resize(static_cast<size_t>(p2b_del), 0.0);
        p2n.resize(static_cast<size_t>(p2n_del), 0.0);
        b2p.resize(static_cast<size_t>(b2p_del), 0.0);
        n2p.resize(static_cast<size_t>(n2p_del), 0.0);

        mt19937 rng(random_device{}());
        uniform_real_distribution<> noise(-0.2, 0.2);
        uniform_real_distribution<> pick(-0.4, 0.4);

        const size_t pick_len = max(size_t(2), b2p.size() / 12);
        for (size_t i = 0; i < p2b.size(); ++i) {
            const StkFloat t =
                static_cast<StkFloat>(i) / static_cast<StkFloat>(p2b.size());
            p2b[i] = (0.5 * t) + noise(rng);
        }
        for (size_t i = 0; i < p2n.size(); ++i) {
            const StkFloat t =
                static_cast<StkFloat>(i) / static_cast<StkFloat>(p2n.size());
            p2n[i] = (0.5 * t) + noise(rng);
        }
        for (size_t i = 0; i < b2p.size(); ++i) {
            const StkFloat t = static_cast<StkFloat>(b2p.size() - i) /
                               static_cast<StkFloat>(b2p.size());
            const StkFloat env = (i < pick_len)
                                     ? (1.0 - (static_cast<StkFloat>(i) /
                                               static_cast<StkFloat>(pick_len)))
                                     : 0.0;
            b2p[i] = (0.5 * t) + (env * pick(rng)) + noise(rng);
        }
        for (size_t i = 0; i < n2p.size(); ++i) {
            const StkFloat t = static_cast<StkFloat>(n2p.size() - i) /
                               static_cast<StkFloat>(n2p.size());
            const StkFloat env = (i < pick_len)
                                     ? (1.0 - (static_cast<StkFloat>(i) /
                                               static_cast<StkFloat>(pick_len)))
                                     : 0.0;
            n2p[i] = (0.5 * t) + (env * pick(rng)) + noise(rng);
        }

        attenuation = pow(10.0, -3.0 / (2.0 * f0 * t60));
    }

    auto tick() -> StkFloat {
        StkFloat p2b_now = 0.0;
        StkFloat b2p_now = 0.0;
        for (size_t i = 0; i < oversample; ++i) {
            const StkFloat p2b_raw = p2b[p2b_idx];
            const StkFloat b2p_raw = b2p[b2p_idx];
            const StkFloat energy = (p2b_raw * p2b_raw) + (b2p_raw * b2p_raw);
            const StkFloat frac = min(tension_coeff * energy, 0.999);

            p2b_now = p2b_raw;
            p2b_now = b2p_filter_lo.tick(p2b_now);
            p2b_now = b2p_filter_hi.tick(p2b_now);

            auto p2n_now = p2n[p2n_idx];
            p2n_now = n2p_filter_lo.tick(p2n_now);
            p2n_now = n2p_filter_hi.tick(p2n_now);

            b2p_now = ((1.0 - frac) * b2p_raw) +
                      (frac * b2p[(b2p_idx + 1) % b2p.size()]);
            for (auto &f : p2n_dispersion) b2p_now = f.tick(b2p_now);

            auto n2p_now = ((1.0 - frac) * n2p[n2p_idx]) +
                           (frac * n2p[(n2p_idx + 1) % n2p.size()]);
            for (auto &f : p2b_dispersion) n2p_now = f.tick(n2p_now);

            p2b[p2b_idx] = n2p_now;
            p2n[p2n_idx] = b2p_now;
            b2p[b2p_idx] = -attenuation * p2b_now;
            n2p[n2p_idx] = -attenuation * p2n_now;

            p2b_idx = (p2b_idx + 1) % p2b.size();
            p2n_idx = (p2n_idx + 1) % p2n.size();
            b2p_idx = (b2p_idx + 1) % b2p.size();
            n2p_idx = (n2p_idx + 1) % n2p.size();
        }

        const StkFloat out = 0.5 * (p2b_now + b2p_now);
        const StkFloat dc_blocked = out - dc_x1 + (0.9995 * dc_y1);
        dc_x1 = out;
        dc_y1 = dc_blocked;
        return dc_blocked;
    }
};

class Distortion {
    const StkFloat G;

    BiQuad lpf = BiQuad::make_lowpass(8'000.0, 0.7);
    BiQuad bpf = BiQuad::make_bandpass(1'600.0, 0.7);
    BiQuad hpf = BiQuad::make_highpass(120.0, 0.7);

  public:
    explicit Distortion(StkFloat G, StkFloat spread = 0.1) : G(G) {
        mt19937 rng(random_device{}());
        uniform_real_distribution<> jitter(-spread, spread);
        lpf = BiQuad::make_lowpass(8'000.0 * (1.0 + jitter(rng)), 0.7);
        bpf = BiQuad::make_bandpass(1'600.0 * (1.0 + jitter(rng)), 0.7);
        hpf = BiQuad::make_highpass(120.0 * (1.0 + jitter(rng)), 0.7);
    }

    auto tick(StkFloat x) -> StkFloat {
        StkFloat y = x;
        y = hpf.tick(y);
        y = bpf.tick(y);
        y = tanh(G * y);
        y = lpf.tick(y);
        return y;
    }
};

struct TickData {
    std::unique_ptr<vector<String>> strings;
    std::unique_ptr<Distortion> distortion;

    TickData(vector<String> &&strings, Distortion distortion)
        : strings(make_unique<vector<String>>(std::move(strings))),
          distortion(make_unique<Distortion>(distortion)) {}
};

auto tick(void *output, void * /* input */, unsigned int n_buffer_frames,
          double /* stream_time */, RtAudioStreamStatus /* status */,
          void *data) -> int {
    auto *tick_data = static_cast<TickData *>(data);
    auto &strings = *tick_data->strings;
    auto &distortion = *tick_data->distortion;
    auto *samples = static_cast<StkFloat *>(output);
    for (size_t i = 0; i < n_buffer_frames; ++i) {
        auto y = 0.0;
        for (auto &string : strings) y += string.tick();
        y = distortion.tick(y);
        for (size_t j = 0; j < n_channels; ++j) {
            *samples = 0.2 * y;
            ++samples;
        }
    }
    return 0;
}

} // namespace

auto main() -> int {
    try {
        auto dac = make_unique<RtAudio>();
        RtAudio::StreamParameters params{
            .deviceId = dac->getDefaultOutputDevice(),
            .nChannels = n_channels,
        };
        RtAudioFormat format = RTAUDIO_FLOAT64;
        unsigned int buffer_size = RT_BUFFER_SIZE;

        TickData data{
            {
                String(82.0 * pow(2.0, 0.0 / 12.0), 10.0, 0.8),
                String(82.0 * pow(2.0, 1.0), 10.0, 0.8),
                String(82.0 * pow(2.0, 16.0 / 12.0), 10.0, 0.8),
                String(82.0 * pow(2.0, 22.0 / 12.0), 10.0, 0.8),
                String(82.0 * pow(2.0, 27.0 / 12.0), 10.0, 0.8),
            },
            Distortion(1.0),
        };

        if (dac->openStream(&params, nullptr, format, sample_rate, &buffer_size,
                            &tick, static_cast<void *>(&data)) != 0U)
            throw StkError(dac->getErrorText());

        if (dac->startStream() != 0U) throw StkError(dac->getErrorText());

        termios oldt{};
        termios newt{};
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        newt.c_cc[VMIN] = 0;
        newt.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        cout << "Pressione ESC para sair...\n";
        while (true) {
            char c = 0;
            if (read(STDIN_FILENO, &c, 1) == 1 && c == '\x1b') break;
            sleep(1);
        }

        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        cout << "Done!\n";
        return 0;
    } catch (StkError &e) {
        cerr << "StkError: " << e.what() << "\n";
    } catch (exception &e) {
        cerr << "Error: " << e.what() << "\n";
    } catch (...) { cerr << "Unknown error!\n"; }
    return 1;
}

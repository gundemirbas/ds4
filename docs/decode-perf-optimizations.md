# DGX Spark (CUDA) Performans Optimizasyon Planı

> decode-perf-tuning branch'inden alınan optimizasyonlar
> Güncelleme: 2026-06-12 (implementasyon tamamlandı)

## Mevcut Durum

DGX Spark (GB10 Grace-Blackwell, sm_121) ~14 t/s generation throughput.
M4 Max ~27 t/s, Blackwell'in potansiyelinin çok altında.

**Kök neden:** CUDA kodu generic kalıp Blackwell'in tensor core'larını
tam kullanamıyor. ROCm kodu AMD-native optimizasyonlarla daha verimli.

---

## Uygulama Durumu

| # | Optimizasyon | Durum | Detay |
|---|-------------|-------|-------|
| 🏆1 | **MMQ/MMVQ Kernel'leri** | ✅ **Uygulandı** | `cuda/mmq/` dizini var, Makefile MMQ_OBJS+INCLUDES ekli, `ds4_cuda_runtime.cuh`'da init+strategy, matmul/MoE dispatch tam. Derleniyor. |
| 🎯2 | **Ayrı MoE Stream** | ✅ **Uygulandı** | `g_moe_stream` + `ds4_cuda_moe_stream()`, tüm MoE launch'lar `moe_stream` kullanıyor. |
| 💾3 | **DGX Spark HBM Cache** | ✅ **Uygulandı** | `DS4_CUDA_SPARK_HBM_CACHE` flag'ı `cuda_model_range_ptr`'de UVA-mapped fallback'leri atlayıp direkt HBM copy'e yöneliyor. Cache limit 120 GiB. |
| 🔬4 | **FP8 KV Cache** | ✅ **Uygulandı** | Packed E4M3 codes + per-64-lane scale + FP32 rotary tail format. `fp8_kv_read()` device helper, `fp8_kv_quantize_kernel` (packed), `fp8_kv_quantize_row_kernel`. Attention decode kernel `comp_fp8`/`comp_scale` pointer'ları alıyor, FP8 yolunda `fp8_kv_read()` kullanıyor. Counter'lar (`g_fp8_kv_read_path_blocks`/`g_fp8_kv_indexed_read_path_blocks`) mevcut. Host-side `ds4_gpu_dsv4_fp8_kv_quantize_pack_tensor()` / `_row_tensor()` entry'leri. |

---

## 🏆 Öncelik 1: MMQ/MMVQ Kernel'leri (llama.cpp vendor)

**Etki:** Prefill ~2.8x, decode ~%5-20 iyileşme
**Zorluk:** Orta
**Durum:** ✅ UYGULANDI

### Ne Alınacak

https://github.com/Entrpi/ds4/tree/decode-perf-tuning/misc altını oku

https://github.com/Entrpi/ds4/tree/decode-perf-tuning branch'indeki `cuda/mmq/` dizini:
- `mmq.cuh` — ana fused-dequant-matmul template'leri (4176 satır)
- `mma.cuh` — CUDA WMMA/PTX mma yardımcıları (1456 satır)
- `vecdotq.cuh` — vector dot-product kernels (1317 satır)
- `mmvq.cu` / `mmvq.cuh` — vector matmul (decode-optimized)
- `quantize.cu` / `quantize.cuh` — activation quantization
- `mmid.cu` / `mmid.cuh` — MoE expert ID helper
- `common.cuh` — ortak CUDA yardımcıları
- `ggml-common.h` — block type tanımları
- `ds4_ggml_stubs.h` / `ds4_ggml_stubs.cu` — ggml shim
- `ds4_mmq.h` / `ds4_mmq.cu` — ds4 C ABI wrapper
- `unary.cuh` — GLU epilogue
- `vendors/cuda.h`

### Yapılacaklar

1. `cuda/mmq/` dizinini decode-perf-tuning'den kopyala
2. Makefile'a MMQ_OBJS ve MMQ_INCLUDES ekle
3. `ds4_cuda.cu`'ya `#include "cuda/mmq/ds4_mmq.h"` ekle
4. `ds4_cuda_runtime.cuh`'a:
   - `ds4_mmq_init()` çağrısı (gpu_init)
   - `g_ds4_use_mmq` state management
   - `ds4_q8_strategy` enum ve selector
   - `ds4_current_stream()` / capture stream altyapısı
5. `ds4_cuda_matmul.cuh`'da `ds4_gpu_matmul_q8_0_tensor`:
   - mmq path: `ds4_mmq_q8_0_dense()` çağrısı
   - mmvq path: `ds4_mmq_q8_0_dense_vec()` (n_tok=1)
   - Strategy selection (mmq > cublas > warp8)
6. `ds4_cuda_moe_launch.cuh`'da MoE dispatch:
   - mmq path: `ds4_mmq_iq2_xxs_moe()`, `ds4_mmq_q2_K_moe()`
   - mmvq decode path: `ds4_mmq_iq2_xxs_moe_vec()`
   - Pair-fused: `ds4_mmq_iq2_xxs_moe_pair()`

### Kod Yapısı

```
cuda/mmq/
├── common.cuh           # vendor: ortak CUDA helpers
├── ds4_ggml_stubs.cu    # shim: ggml tip tanımları, pool
├── ds4_ggml_stubs.h     # shim: ggml_type enum, macros
├── ds4_mmq.cu           # wrapper: ds4 C ABI entry'leri
├── ds4_mmq.h            # wrapper: public API
├── ggml-common.h        # vendor: block type struct'ları
├── ggml-cuda.h          # redirect -> ds4_ggml_stubs.h
├── ggml-impl.h          # redirect -> ds4_ggml_stubs.h
├── ggml.h               # redirect -> ds4_ggml_stubs.h
├── mma.cuh              # vendor: WMMA helpers
├── mmid.cu              # vendor: MoE ID helper
├── mmid.cuh             # vendor: MoE ID header
├── mmq.cuh              # vendor: ANA fused-dequant-matmul
├── mmvq.cu              # vendor: vector matmul (decode)
├── mmvq.cuh             # vendor: vector matmul header
├── quantize.cu          # vendor: Q8_1 quantization
├── quantize.cuh         # vendor: quantization header
├── unary.cuh            # vendor: unary ops (GLU epilogue)
├── VENDOR.md            # vendor bilgisi
└── vendors/
    └── cuda.h           # vendor: CUDA header
```

---

## 🎯 Öncelik 2: Ayrı MoE Stream (Async Expert Pipeline)

**Etki:** Decode ~%5-15 iyileşme
**Zorluk:** Düşük
**Durum:** ✅ UYGULANDI

### Ne Alınacak

```c
static cudaStream_t g_moe_stream = NULL;

static cudaStream_t ds4_cuda_moe_stream(void) {
    if (!g_moe_stream) {
        cudaError_t ge = cudaStreamCreateWithFlags(&g_moe_stream, cudaStreamNonBlocking);
        // error handling
    }
    return g_moe_stream;
}
```

### Yapılacaklar

1. `ds4_cuda_runtime.cuh`'a `g_moe_stream` + `ds4_cuda_moe_stream()` ekle
2. `ds4_cuda_moe_launch.cuh`'daki tüm MoE kernel launch'larına `moe_stream` parametresi ekle
3. `ds4_cuda_matmul.cuh`'daki mmq MoE çağrılarına `moe_stream` parametresi ekle
4. Stream hazard yönetimi (sync pre/post)

---

## 💾 Öncelik 3: DGX Spark HBM Cache

**Etki:** Spark ~%10-30 iyileşme
**Zorluk:** Düşük
**Durum:** ✅ UYGULANDI

### Ne Alınacak

```makefile
CUDA_SPARK_FLAGS := -DDS4_CUDA_SPARK_HBM_CACHE=1

cuda-spark:
	$(MAKE) -B ... CUDA_ARCH= \
	    CFLAGS="$(CFLAGS) $(CUDA_SPARK_FLAGS)" \
	    NVCCFLAGS="$(NVCCFLAGS) $(CUDA_SPARK_FLAGS)"
```

### Yapılacaklar

1. Makefile'a `CUDA_SPARK_FLAGS` tanımı
2. `cuda-spark` hedefine flag ekle
3. `ds4_cuda_runtime.cuh`'da `DS4_CUDA_SPARK_HBM_CACHE` guard'ı ile HBM yönetimi

---

## 🔬 Öncelik 4: FP8 KV Cache (Opp C Phase 1A)

**Etki:** Uzun context ~%10-20
**Zorluk:** Yüksek
**Durum:** ✅ UYGULANDI

### Ne Alınacak

- Packed E4M3 codes + per-64-lane scale + FP32 rotary tail
- `comp_cache_fp8`, `comp_scale` tensor'ları
- `ds4_gpu_dsv4_fp8_kv_quantize_tensor()` / `_row_tensor()`
- Attention kernel FP8 okuma yolu

---

## ✅ Zaten Optimize Olanlar (Değişiklik Gerekmez)

| Özellik | Dosya | Durum |
|---------|-------|-------|
| `__vcmpne4`/`__vsub4` IQ2 intrinsics | `ds4_cuda_q8_K.cuh` | ✅ Var |
| WMMA indexer kernel'leri | `ds4_cuda_indexer.cuh` | ✅ Var |
| F16 GEMM via cuBLAS | `ds4_cuda_matmul.cuh` | ✅ Var |
| Q8→F16 weight cache | `ds4_cuda_runtime.cuh` | ✅ Var |

---

## Uygulama Sırası

```
Gün 1: MMQ vendor + Makefile + derleme
Gün 1: Ayrı MoE Stream
Gün 1: DGX Spark HBM Cache
---
Gün 2: MMQ matmul entegrasyonu
Gün 2: MMQ MoE entegrasyonu
Gün 2: Test + doğrulama
---
Gün 3+: FP8 KV Cache
```

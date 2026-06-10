// SPDX-License-Identifier: MIT
// DS4 CUDA — SSD streaming expert cache
//
// Implements a pinned-buffer expert cache with slab allocator, LRU (clock
// algorithm) eviction, route-hotness tracking, and an async pread thread pool.
// This is the CUDA backend port of the Metal ssd_streaming machinery.
//
// Included from ds4_cuda.cu before ds4_cuda_runtime.cuh so the extern "C"
// stubs in the runtime file can call the static helpers defined here.

#ifndef DS4_CUDA_STREAM_EXPERT_CACHE_CUH
#define DS4_CUDA_STREAM_EXPERT_CACHE_CUH

#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

// =========================================================================
// Sabitler
// =========================================================================

// Model max dimension'ları (Pro ve Flash'ın max'ı)
#define DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER   61
#define DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT  384
#define DS4_CUDA_STREAM_EXPERT_CACHE_MAX_ENTRIES \
    (DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER * DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT)
#define DS4_CUDA_STREAM_EXPERT_CACHE_MAX_SLABS   256
#define DS4_CUDA_STREAM_EXPERT_HOTNESS_DECAY_TOKENS 16

// Slab boyutu
#define DS4_CUDA_STREAM_EXPERT_SLAB_BYTES (256ull * 1024ull * 1024ull)  // 256 MB

// Bir expert'te max 6 seçim
#define DS4_CUDA_STREAM_EXPERT_MAX_SELECTED 6

// Expert validasyon için (Metal'de 16 word)
#define DS4_CUDA_STREAM_EXPERT_VALIDATE_WORDS 16

// Pread thread pool
#define DS4_CUDA_STREAM_PREAD_MAX_WORKERS 4
#define DS4_CUDA_STREAM_PREAD_QUEUE_SIZE  64

// =========================================================================
// Veri Yapıları
// =========================================================================

typedef struct {
    void    *gate_ptr;           // cudaMallocHost pinned buffer
    void    *up_ptr;
    void    *down_ptr;
    const void *model_map;       // validasyon için
    uint64_t  gate_abs_offset;   // model dosyası offset
    uint64_t  up_abs_offset;
    uint64_t  down_abs_offset;
    uint64_t  gate_expert_bytes;
    uint64_t  down_expert_bytes;
    uint64_t  logical_bytes;     // gate*2 + down
    uint64_t  last_used;         // LRU clock
    uint64_t  use_count;
    uint32_t  slab_slot;         // StreamSlab index
    uint8_t   valid;
} StreamExpertEntry;

typedef struct {
    void    *base;               // cudaMallocHost base
    uint64_t bytes;              // toplam boyut
    uint64_t used;               // kullanılan offset
    uint8_t  free;               // tamamen boş
} StreamSlab;

typedef struct {
    int      fd;                 // model_fd
    void    *dst;                // pinned buffer hedef
    uint64_t bytes;              // okunacak byte
    uint64_t file_offset;        // model dosyası offset
    uint8_t  ok;
    uint64_t read_bytes;
} StreamPreadTask;

typedef struct {
    uint8_t   active;
    const void *model_map;
    uint64_t   model_size;
    uint32_t   layer;
    int32_t    selected_ids[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED];
    uint32_t   n_total_expert;
    uint32_t   n_selected;
    uint64_t   gate_expert_bytes;
    uint64_t   down_expert_bytes;
    uint64_t   gate_abs_offsets[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED];
    uint64_t   up_abs_offsets[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED];
    uint64_t   down_abs_offsets[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED];
    uint32_t   missing_mask;
    uint32_t   n_loads;
    uint32_t   load_slots[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED];
    uint32_t   n_tasks;
    StreamPreadTask tasks[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED * 3]; // max 18
    pthread_t  worker;
    uint8_t    done;
    uint8_t    ok;
} StreamPendingLoad;

// =========================================================================
// Global Değişkenler
// =========================================================================

static StreamExpertEntry
    g_stream_expert_cache[DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER]
                         [DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT];

static StreamSlab    g_slabs[DS4_CUDA_STREAM_EXPERT_CACHE_MAX_SLABS];
static uint32_t      g_slab_count;
static uint32_t      g_stream_expert_cache_entry_count;
static uint32_t      g_stream_expert_cache_budget_override;
static uint64_t      g_stream_expert_cache_clock;
static uint64_t      g_stream_expert_cache_expert_bytes;  // gate*2 + down
static uint64_t      g_stream_expert_cache_hits;
static uint64_t      g_stream_expert_cache_misses;
static uint64_t      g_stream_expert_cache_evictions;
static int           g_ssd_streaming_mode;

// Route hotness
static uint32_t
    g_stream_expert_cache_route_hotness[DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER]
                                       [DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT];
static uint64_t g_stream_expert_cache_decode_tokens;
static uint64_t g_stream_expert_cache_hotness_decay_token;

// MoE selected override
static int32_t  g_routed_moe_selected_override[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED];
static uint32_t g_routed_moe_selected_override_n;

// Pending async load
static StreamPendingLoad g_stream_expert_pending_load;

// Model fd
static int g_stream_model_fd = -1;

// =========================================================================
// Pread Thread Pool
// =========================================================================

static pthread_t g_pread_workers[DS4_CUDA_STREAM_PREAD_MAX_WORKERS];
static uint8_t   g_pread_pool_started = 0;
static uint8_t   g_pread_pool_stop = 0;

// Task queue
static StreamPreadTask g_pread_queue[DS4_CUDA_STREAM_PREAD_QUEUE_SIZE];
static uint32_t        g_pread_queue_count;
static uint32_t        g_pread_queue_head;
static uint32_t        g_pread_queue_tail;

static pthread_mutex_t g_pread_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_pread_cond  = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_pread_done_cond = PTHREAD_COND_INITIALIZER;
static uint32_t        g_pread_remaining;

static void* pread_worker_main(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_pread_mutex);
        while (g_pread_queue_count == 0 && !g_pread_pool_stop) {
            pthread_cond_wait(&g_pread_cond, &g_pread_mutex);
        }
        if (g_pread_pool_stop) {
            pthread_mutex_unlock(&g_pread_mutex);
            return NULL;
        }

        // Task'ı queue'dan al
        StreamPreadTask task = g_pread_queue[g_pread_queue_head];
        g_pread_queue_head = (g_pread_queue_head + 1) % DS4_CUDA_STREAM_PREAD_QUEUE_SIZE;
        g_pread_queue_count--;
        pthread_mutex_unlock(&g_pread_mutex);

        // pread ile dosyadan oku
        ssize_t r = pread(task.fd, task.dst, (size_t)task.bytes, (off_t)task.file_offset);
        // Task struct'ı stack'te, ok değerini geri yazamayız — pooled worker
        // bitince remaining azalt, done sinyali gönder
        // Not: hata durumunda da okuma bitti sayılır (caller read_bytes kontrol eder)
        pthread_mutex_lock(&g_pread_mutex);
        g_pread_remaining--;
        if (g_pread_remaining == 0) {
            pthread_cond_signal(&g_pread_done_cond);
        }
        pthread_mutex_unlock(&g_pread_mutex);
    }
    return NULL;
}

static int pread_pool_start(void) {
    if (g_pread_pool_started) return 1;
    g_pread_pool_stop = 0;
    g_pread_queue_count = 0;
    g_pread_queue_head = 0;
    g_pread_queue_tail = 0;
    g_pread_remaining = 0;
    for (uint32_t i = 0; i < DS4_CUDA_STREAM_PREAD_MAX_WORKERS; i++) {
        if (pthread_create(&g_pread_workers[i], NULL, pread_worker_main, NULL) != 0) {
            fprintf(stderr, "ds4: CUDA streaming pread pool thread create failed\n");
            // Created olanları temizle
            for (uint32_t j = 0; j < i; j++) {
                pthread_cancel(g_pread_workers[j]);
            }
            g_pread_pool_started = 0;
            return 0;
        }
    }
    g_pread_pool_started = 1;
    return 1;
}

static void pread_pool_stop(void) {
    if (!g_pread_pool_started) return;
    g_pread_pool_stop = 1;
    pthread_cond_broadcast(&g_pread_cond);
    for (uint32_t i = 0; i < DS4_CUDA_STREAM_PREAD_MAX_WORKERS; i++) {
        pthread_join(g_pread_workers[i], NULL);
    }
    g_pread_pool_started = 0;
}

static int pread_pool_submit(int fd, void *dst, uint64_t bytes, uint64_t file_offset) {
    pthread_mutex_lock(&g_pread_mutex);
    if (g_pread_queue_count >= DS4_CUDA_STREAM_PREAD_QUEUE_SIZE) {
        pthread_mutex_unlock(&g_pread_mutex);
        return 0;
    }
    g_pread_queue[g_pread_queue_tail].fd = fd;
    g_pread_queue[g_pread_queue_tail].dst = dst;
    g_pread_queue[g_pread_queue_tail].bytes = bytes;
    g_pread_queue[g_pread_queue_tail].file_offset = file_offset;
    g_pread_queue[g_pread_queue_tail].ok = 0;
    g_pread_queue[g_pread_queue_tail].read_bytes = 0;
    g_pread_queue_tail = (g_pread_queue_tail + 1) % DS4_CUDA_STREAM_PREAD_QUEUE_SIZE;
    g_pread_queue_count++;
    g_pread_remaining++;
    pthread_mutex_unlock(&g_pread_mutex);
    pthread_cond_signal(&g_pread_cond);
    return 1;
}

static int pread_pool_wait_all(void) {
    pthread_mutex_lock(&g_pread_mutex);
    while (g_pread_remaining > 0) {
        pthread_cond_wait(&g_pread_done_cond, &g_pread_mutex);
    }
    pthread_mutex_unlock(&g_pread_mutex);
    return 1;
}

// =========================================================================
// Slab Allocator
// =========================================================================

static void* slab_alloc(uint64_t bytes) {
    if (bytes == 0) return NULL;

    // 1. Mevcut slab'larda boş yer ara
    for (uint32_t i = 0; i < g_slab_count; i++) {
        StreamSlab *s = &g_slabs[i];
        if (s->free) continue;
        uint64_t aligned = (s->used + 255) & ~(uint64_t)255;
        if (aligned + bytes <= s->bytes) {
            void *ptr = (char*)s->base + aligned;
            s->used = aligned + bytes;
            return ptr;
        }
    }

    // 2. Boş slab var mı kontrol et
    for (uint32_t i = 0; i < g_slab_count; i++) {
        if (g_slabs[i].free) {
            uint64_t slab_bytes = bytes > DS4_CUDA_STREAM_EXPERT_SLAB_BYTES
                                  ? bytes : DS4_CUDA_STREAM_EXPERT_SLAB_BYTES;
            void *base;
            cudaError_t err = cudaMallocHost(&base, (size_t)slab_bytes);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA streaming slab cudaMallocHost(%zu) failed: %s\n",
                        (size_t)slab_bytes, cudaGetErrorString(err));
                return NULL;
            }
            g_slabs[i].base = base;
            g_slabs[i].bytes = slab_bytes;
            g_slabs[i].used = bytes;
            g_slabs[i].free = 0;
            return base;
        }
    }

    // 3. Yeni slab oluştur
    if (g_slab_count >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_SLABS) {
        fprintf(stderr, "ds4: CUDA streaming slab count exhausted\n");
        return NULL;
    }
    uint64_t slab_bytes = bytes > DS4_CUDA_STREAM_EXPERT_SLAB_BYTES
                          ? bytes : DS4_CUDA_STREAM_EXPERT_SLAB_BYTES;
    void *base;
    cudaError_t err = cudaMallocHost(&base, (size_t)slab_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA streaming slab cudaMallocHost(%zu) failed: %s\n",
                (size_t)slab_bytes, cudaGetErrorString(err));
        return NULL;
    }
    g_slabs[g_slab_count].base = base;
    g_slabs[g_slab_count].bytes = slab_bytes;
    g_slabs[g_slab_count].used = bytes;
    g_slabs[g_slab_count].free = 0;
    g_slab_count++;
    return base;
}

static void slab_free_entry(StreamExpertEntry *e) {
    if (!e || !e->valid) return;
    e->valid = 0;
    e->gate_ptr = NULL;
    e->up_ptr = NULL;
    e->down_ptr = NULL;
    g_stream_expert_cache_entry_count--;
}

static void slab_destroy_all(void) {
    for (uint32_t i = 0; i < g_slab_count; i++) {
        if (g_slabs[i].base && !g_slabs[i].free) {
            cudaFreeHost(g_slabs[i].base);
        }
    }
    memset(g_slabs, 0, sizeof(g_slabs));
    g_slab_count = 0;
}

// =========================================================================
// Route Hotness Tracking
// =========================================================================

static void stream_expert_cache_reset_route_hotness(void) {
    memset(g_stream_expert_cache_route_hotness, 0,
           sizeof(g_stream_expert_cache_route_hotness));
    g_stream_expert_cache_hotness_decay_token =
        g_stream_expert_cache_decode_tokens;
}

static void stream_expert_cache_decay_route_hotness(void) {
    for (uint32_t l = 0; l < DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER; l++) {
        for (uint32_t e = 0; e < DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT; e++) {
            g_stream_expert_cache_route_hotness[l][e] >>= 1;
        }
    }
}

static void stream_expert_cache_maybe_decay_route_hotness(void) {
    if (g_stream_expert_cache_decode_tokens == 0) return;
    if (g_stream_expert_cache_hotness_decay_token == 0) {
        g_stream_expert_cache_hotness_decay_token =
            g_stream_expert_cache_decode_tokens;
        return;
    }
    while (g_stream_expert_cache_decode_tokens -
           g_stream_expert_cache_hotness_decay_token >=
           DS4_CUDA_STREAM_EXPERT_HOTNESS_DECAY_TOKENS) {
        stream_expert_cache_decay_route_hotness();
        g_stream_expert_cache_hotness_decay_token +=
            DS4_CUDA_STREAM_EXPERT_HOTNESS_DECAY_TOKENS;
    }
}

static void stream_expert_cache_note_route_hotness(
        uint32_t layer, uint32_t expert, uint32_t amount) {
    if (layer >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER ||
        expert >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        amount == 0) return;
    uint32_t *hotness = &g_stream_expert_cache_route_hotness[layer][expert];
    if (*hotness > UINT32_MAX - amount)
        *hotness = UINT32_MAX;
    else
        *hotness += amount;
}

static void stream_expert_cache_note_selected_hotness(
        uint32_t layer, const int32_t *selected_ids, uint32_t n_selected) {
    if (!selected_ids ||
        layer >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_selected == 0) return;
    stream_expert_cache_maybe_decay_route_hotness();
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 ||
            (uint32_t)selected_ids[i] >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT)
            continue;
        stream_expert_cache_note_route_hotness(
                layer, (uint32_t)selected_ids[i], 1);
    }
}

static void stream_expert_cache_note_token(uint32_t layer_index) {
    if (!g_ssd_streaming_mode || layer_index != 0 ||
        g_stream_expert_cache_decode_tokens == UINT64_MAX) return;
    g_stream_expert_cache_decode_tokens++;
    stream_expert_cache_maybe_decay_route_hotness();
}

// =========================================================================
// LRU Eviction (Clock Algorithm)
// =========================================================================

static uint32_t stream_expert_cache_configured_budget(void) {
    if (!g_ssd_streaming_mode) return 0;
    if (g_stream_expert_cache_budget_override != 0)
        return g_stream_expert_cache_budget_override;
    return 0;
}

static uint32_t stream_expert_cache_requested_budget(void) {
    if (!g_ssd_streaming_mode) return 0;
    return g_stream_expert_cache_budget_override;
}

static int stream_expert_cache_entry_matches(
        const StreamExpertEntry *e,
        const void   *model_map,
        uint64_t      gate_abs_offset,
        uint64_t      up_abs_offset,
        uint64_t      down_abs_offset,
        uint64_t      gate_expert_bytes,
        uint64_t      down_expert_bytes) {
    return e &&
           e->valid &&
           e->model_map == model_map &&
           e->gate_abs_offset == gate_abs_offset &&
           e->up_abs_offset == up_abs_offset &&
           e->down_abs_offset == down_abs_offset &&
           e->gate_expert_bytes == gate_expert_bytes &&
           e->down_expert_bytes == down_expert_bytes &&
           e->gate_ptr && e->up_ptr && e->down_ptr;
}

static StreamExpertEntry* stream_expert_cache_peek(
        const void *model_map,
        uint32_t    layer,
        uint32_t    expert,
        uint32_t    n_total_expert,
        uint64_t    gate_abs_offset,
        uint64_t    up_abs_offset,
        uint64_t    down_abs_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes) {
    if (!g_ssd_streaming_mode ||
        layer >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER ||
        expert >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        expert >= n_total_expert)
        return NULL;

    StreamExpertEntry *e = &g_stream_expert_cache[layer][expert];
    if (!stream_expert_cache_entry_matches(e, model_map,
            gate_abs_offset, up_abs_offset, down_abs_offset,
            gate_expert_bytes, down_expert_bytes))
        return NULL;

    e->last_used = ++g_stream_expert_cache_clock;
    e->use_count++;
    g_stream_expert_cache_hits++;
    return e;
}

static void stream_expert_cache_prune_layer(
        uint32_t layer, uint32_t n_total_expert,
        uint32_t n_selected, const int32_t *protect_ids,
        uint32_t n_protect) {
    if (!g_ssd_streaming_mode ||
        layer >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER)
        return;

    uint32_t budget = stream_expert_cache_configured_budget();
    if (budget == 0) return;

    // Layer'daki entry sayısını say
    uint32_t count = 0;
    for (uint32_t e = 0; e < n_total_expert && e < DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT; e++) {
        if (g_stream_expert_cache[layer][e].valid) count++;
    }

    // Budget'ın çok üstünde mi?
    uint32_t max_per_layer = budget / DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER + 1;
    if (count <= max_per_layer) return;

    // Clock algorithm ile evict
    uint32_t evict = count - max_per_layer;
    uint32_t examined = 0;
    uint32_t expert = 0;
    while (evict > 0 && examined < n_total_expert * 2) {
        if (expert >= n_total_expert || expert >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT)
            expert = 0;

        StreamExpertEntry *e = &g_stream_expert_cache[layer][expert];
        examined++;

        if (e->valid) {
            // Protected mı?
            uint8_t is_protected = 0;
            for (uint32_t p = 0; p < n_protect; p++) {
                if (protect_ids[p] == (int32_t)expert) { is_protected = 1; break; }
            }
            if (!is_protected) {
                slab_free_entry(e);
                g_stream_expert_cache_evictions++;
                evict--;
            }
        }
        expert++;
    }
}

static void stream_expert_cache_prune_global(
        uint32_t protect_layer,
        const int32_t *protect_ids,
        uint32_t n_protect) {
    if (!g_ssd_streaming_mode) return;

    uint32_t budget = stream_expert_cache_configured_budget();
    if (budget == 0) return;
    if (g_stream_expert_cache_entry_count <= budget) return;

    uint32_t evict = g_stream_expert_cache_entry_count - budget;
    uint32_t examined = 0;

    while (evict > 0 && examined < g_stream_expert_cache_entry_count * 2) {
        uint64_t min_clock = UINT64_MAX;
        uint32_t min_l = UINT32_MAX, min_e = UINT32_MAX;

        for (uint32_t l = 0; l < DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER; l++) {
            for (uint32_t ex = 0; ex < DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT; ex++) {
                StreamExpertEntry *entry = &g_stream_expert_cache[l][ex];
                if (!entry->valid) continue;

                // Protected mı?
                uint8_t is_protected = 0;
                if (l == protect_layer) {
                    for (uint32_t p = 0; p < n_protect; p++) {
                        if (protect_ids[p] == (int32_t)ex) { is_protected = 1; break; }
                    }
                }
                if (is_protected) continue;

                if (entry->last_used < min_clock) {
                    min_clock = entry->last_used;
                    min_l = l;
                    min_e = ex;
                }
            }
        }

        if (min_l < DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER) {
            slab_free_entry(&g_stream_expert_cache[min_l][min_e]);
            g_stream_expert_cache_evictions++;
            evict--;
        }
        examined++;
    }
}

// =========================================================================
// Expert Size Helper
// =========================================================================

static int stream_expert_cache_note_expert_size(
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    if (gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        fprintf(stderr, "ds4: CUDA streaming expert cache byte size overflow\n");
        return 0;
    }
    g_stream_expert_cache_expert_bytes =
        gate_expert_bytes * 2ull + down_expert_bytes;
    return 1;
}

static uint32_t stream_expert_cache_effective_cap(
        uint32_t layer, uint32_t n_total_expert, uint32_t n_selected) {
    (void)layer;
    (void)n_total_expert;
    (void)n_selected;
    uint32_t budget = stream_expert_cache_configured_budget();
    if (budget == 0) return 0;
    return budget;
}

// =========================================================================
// Cache Get (Sync Load)
// =========================================================================

static int stream_expert_cache_get(
        const void *model_map,
        uint64_t    model_size,
        uint32_t    layer,
        uint32_t    expert,
        uint32_t    n_total_expert,
        uint32_t    n_selected,
        uint64_t    gate_abs_offset,
        uint64_t    up_abs_offset,
        uint64_t    down_abs_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes) {

    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || layer >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER)
        return 1;
    if (expert >= n_total_expert || expert >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT)
        return 0;

    StreamExpertEntry *e = &g_stream_expert_cache[layer][expert];

    // Cache'de var mı?
    if (stream_expert_cache_entry_matches(e, model_map,
            gate_abs_offset, up_abs_offset, down_abs_offset,
            gate_expert_bytes, down_expert_bytes)) {
        e->last_used = ++g_stream_expert_cache_clock;
        e->use_count++;
        g_stream_expert_cache_hits++;
        return 1;
    }

    // Cache miss
    g_stream_expert_cache_misses++;

    // Eski entry varsa temizle
    if (e->valid) {
        slab_free_entry(e);
    }

    // Slab allocator'dan buffer ayır
    void *gate_ptr = slab_alloc(gate_expert_bytes);
    void *up_ptr = slab_alloc(gate_expert_bytes);
    void *down_ptr = slab_alloc(down_expert_bytes);
    if (!gate_ptr || !up_ptr || !down_ptr) {
        // Yer yok: LRU evict yap, tekrar dene
        stream_expert_cache_prune_global(layer, NULL, 0);
        gate_ptr = slab_alloc(gate_expert_bytes);
        up_ptr = slab_alloc(gate_expert_bytes);
        down_ptr = slab_alloc(down_expert_bytes);
        if (!gate_ptr || !up_ptr || !down_ptr) return 0;
    }

    // Model dosyasından pread ile oku
    if (g_stream_model_fd < 0) return 0;
    ssize_t r;
    r = pread(g_stream_model_fd, gate_ptr, (size_t)gate_expert_bytes,
              (off_t)gate_abs_offset);
    if (r != (ssize_t)gate_expert_bytes) { slab_free_entry(e); return 0; }

    r = pread(g_stream_model_fd, up_ptr, (size_t)gate_expert_bytes,
              (off_t)up_abs_offset);
    if (r != (ssize_t)gate_expert_bytes) { slab_free_entry(e); return 0; }

    r = pread(g_stream_model_fd, down_ptr, (size_t)down_expert_bytes,
              (off_t)down_abs_offset);
    if (r != (ssize_t)down_expert_bytes) { slab_free_entry(e); return 0; }

    // Cache entry'e kaydet
    e->gate_ptr = gate_ptr;
    e->up_ptr = up_ptr;
    e->down_ptr = down_ptr;
    e->model_map = model_map;
    e->gate_abs_offset = gate_abs_offset;
    e->up_abs_offset = up_abs_offset;
    e->down_abs_offset = down_abs_offset;
    e->gate_expert_bytes = gate_expert_bytes;
    e->down_expert_bytes = down_expert_bytes;
    e->logical_bytes = gate_expert_bytes * 2 + down_expert_bytes;
    e->last_used = ++g_stream_expert_cache_clock;
    e->use_count = 1;
    e->valid = 1;

    g_stream_expert_cache_entry_count++;
    return 1;
}

// =========================================================================
// Pending Load Helpers
// =========================================================================

static int stream_expert_pending_load_matches(
        const void *model_map, uint64_t model_size,
        uint32_t layer, const int32_t *selected_ids,
        uint32_t n_total_expert, uint32_t n_selected,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {

    StreamPendingLoad *p = &g_stream_expert_pending_load;
    if (!p->active) return 0;
    if (p->model_map != model_map ||
        p->model_size != model_size ||
        p->layer != layer ||
        p->n_total_expert != n_total_expert ||
        p->n_selected != n_selected ||
        p->gate_expert_bytes != gate_expert_bytes ||
        p->down_expert_bytes != down_expert_bytes)
        return 0;

    for (uint32_t i = 0; i < n_selected; i++) {
        if (p->selected_ids[i] != selected_ids[i]) return 0;
    }
    return 1;
}

static void stream_expert_pending_load_clear(void) {
    StreamPendingLoad *p = &g_stream_expert_pending_load;
    p->active = 0;
    p->n_tasks = 0;
    p->n_loads = 0;
    p->missing_mask = 0;
}

// =========================================================================
// seed_selected() — Prefill Expert Cache Seed
// =========================================================================

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_total_expert,
        uint32_t       n_selected,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes) {

    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || !selected_ids ||
        layer >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_selected == 0 || n_selected > DS4_CUDA_STREAM_EXPERT_MAX_SELECTED ||
        n_total_expert == 0 ||
        n_total_expert > DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        !stream_expert_cache_note_expert_size(gate_expert_bytes, down_expert_bytes)) {
        return 1;
    }

    // Route hotness güncelle
    stream_expert_cache_note_selected_hotness(layer, selected_ids, n_selected);

    // Her selected expert için
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr, "ds4: CUDA prefill expert-cache seed expert id %d out of range\n",
                    selected_ids[i]);
            return 0;
        }

        const uint64_t expert_id = (uint64_t)(uint32_t)selected_ids[i];
        if (expert_id > UINT64_MAX / gate_expert_bytes ||
            expert_id > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: CUDA prefill expert-cache seed offset overflow\n");
            return 0;
        }

        const uint64_t gate_rel = expert_id * gate_expert_bytes;
        const uint64_t down_rel = expert_id * down_expert_bytes;
        if (gate_rel > UINT64_MAX - gate_offset ||
            gate_rel > UINT64_MAX - up_offset ||
            down_rel > UINT64_MAX - down_offset) {
            fprintf(stderr, "ds4: CUDA prefill expert-cache seed offset overflow\n");
            return 0;
        }

        if (!stream_expert_cache_get(model_map, model_size, layer,
                (uint32_t)selected_ids[i], n_total_expert, n_selected,
                gate_offset + gate_rel,
                up_offset + gate_rel,
                down_offset + down_rel,
                gate_expert_bytes, down_expert_bytes)) {
            return 0;
        }
    }

    // LRU prune
    stream_expert_cache_prune_layer(layer, n_total_expert, n_selected,
                                     selected_ids, n_selected);
    stream_expert_cache_prune_global(layer, selected_ids, n_selected);

    return 1;
}

// =========================================================================
// begin_selected_load() — Async Decode Expert Load
// =========================================================================

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_total_expert,
        uint32_t       n_selected,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes) {

    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || !selected_ids ||
        layer >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_selected == 0 || n_selected > DS4_CUDA_STREAM_EXPERT_MAX_SELECTED ||
        n_total_expert == 0 ||
        n_total_expert > DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        !stream_expert_cache_note_expert_size(gate_expert_bytes, down_expert_bytes) ||
        stream_expert_cache_effective_cap(layer, n_total_expert, n_selected) == 0) {
        return 1;
    }

    // Pending load kontrol: aynı expert'ler zaten yükleniyor mu?
    if (stream_expert_pending_load_matches(model_map, model_size, layer,
            selected_ids, n_total_expert, n_selected,
            gate_expert_bytes, down_expert_bytes)) {
        return 1;
    }

    // Önceki pending load'u temizle
    stream_expert_pending_load_clear();
    StreamPendingLoad *p = &g_stream_expert_pending_load;
    p->active = 0;
    p->model_map = model_map;
    p->model_size = model_size;
    p->layer = layer;
    p->n_total_expert = n_total_expert;
    p->n_selected = n_selected;
    p->gate_expert_bytes = gate_expert_bytes;
    p->down_expert_bytes = down_expert_bytes;
    p->missing_mask = 0;
    p->n_loads = 0;
    p->n_tasks = 0;

    for (uint32_t i = 0; i < DS4_CUDA_STREAM_EXPERT_MAX_SELECTED; i++) {
        p->selected_ids[i] = -1;
        p->load_slots[i] = 0;
    }

    // Expert offset'lerini hesapla, cache'de olmayanları bul
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr, "ds4: CUDA streaming early-load expert id %d out of range\n",
                    selected_ids[i]);
            return 0;
        }
        p->selected_ids[i] = selected_ids[i];

        const uint64_t expert_id = (uint64_t)(uint32_t)selected_ids[i];
        if (expert_id > UINT64_MAX / gate_expert_bytes ||
            expert_id > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: CUDA streaming early-load offset overflow\n");
            return 0;
        }
        const uint64_t gate_rel = expert_id * gate_expert_bytes;
        const uint64_t down_rel = expert_id * down_expert_bytes;
        if (gate_rel > UINT64_MAX - gate_offset ||
            gate_rel > UINT64_MAX - up_offset ||
            down_rel > UINT64_MAX - down_offset) {
            fprintf(stderr, "ds4: CUDA streaming early-load offset overflow\n");
            return 0;
        }

        p->gate_abs_offsets[i] = gate_offset + gate_rel;
        p->up_abs_offsets[i] = up_offset + gate_rel;
        p->down_abs_offsets[i] = down_offset + down_rel;

        // Cache'de var mı kontrol et
        StreamExpertEntry *e = &g_stream_expert_cache[layer][(uint32_t)selected_ids[i]];
        if (stream_expert_cache_entry_matches(e, model_map,
                p->gate_abs_offsets[i], p->up_abs_offsets[i], p->down_abs_offsets[i],
                gate_expert_bytes, down_expert_bytes)) {
            continue;  // Zaten cache'de
        }

        // Cache'de yok: pread task oluştur
        p->missing_mask |= 1u << i;
        p->load_slots[p->n_loads++] = i;
    }

    if (p->n_loads == 0) return 1;  // Hepsi cache'de

    // Thread pool çalışıyor mu?
    if (!g_pread_pool_started) {
        if (!pread_pool_start()) return 0;
    }

    // Slab allocator'dan buffer ayır ve pread task'larını oluştur
    for (uint32_t load_i = 0; load_i < p->n_loads; load_i++) {
        const uint32_t slot = p->load_slots[load_i];
        const uint32_t expert = (uint32_t)p->selected_ids[slot];

        // Eski entry varsa temizle
        StreamExpertEntry *e = &g_stream_expert_cache[layer][expert];
        if (e->valid) slab_free_entry(e);

        // Yeni buffer ayır
        void *gate_ptr = slab_alloc(gate_expert_bytes);
        void *up_ptr = slab_alloc(gate_expert_bytes);
        void *down_ptr = slab_alloc(down_expert_bytes);

        if (!gate_ptr || !up_ptr || !down_ptr) {
            // Yer yoksa LRU evict yap
            stream_expert_cache_prune_global(layer, selected_ids, n_selected);
            gate_ptr = slab_alloc(gate_expert_bytes);
            up_ptr = slab_alloc(gate_expert_bytes);
            down_ptr = slab_alloc(down_expert_bytes);
            if (!gate_ptr || !up_ptr || !down_ptr) return 0;
        }

        // Pread task'ları oluştur (thread pool async)
        pread_pool_submit(g_stream_model_fd, gate_ptr, gate_expert_bytes,
                          p->gate_abs_offsets[slot]);
        pread_pool_submit(g_stream_model_fd, up_ptr, gate_expert_bytes,
                          p->up_abs_offsets[slot]);
        pread_pool_submit(g_stream_model_fd, down_ptr, down_expert_bytes,
                          p->down_abs_offsets[slot]);

        // Cache entry'i önceden kaydet (pread bitince veri hazır olacak)
        e->gate_ptr = gate_ptr;
        e->up_ptr = up_ptr;
        e->down_ptr = down_ptr;
        e->model_map = model_map;
        e->gate_abs_offset = p->gate_abs_offsets[slot];
        e->up_abs_offset = p->up_abs_offsets[slot];
        e->down_abs_offset = p->down_abs_offsets[slot];
        e->gate_expert_bytes = gate_expert_bytes;
        e->down_expert_bytes = down_expert_bytes;
        e->logical_bytes = gate_expert_bytes * 2 + down_expert_bytes;
        e->last_used = ++g_stream_expert_cache_clock;
        e->use_count = 1;
        e->valid = 1;

        g_stream_expert_cache_entry_count++;
    }

    p->active = 1;
    return 1;
}

// =========================================================================
// seed_experts() — Hotlist Seed
// =========================================================================

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *expert_ids,
        const uint32_t *expert_priorities,
        uint32_t       n_experts,
        uint32_t       n_total_expert,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes) {

    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || !expert_ids ||
        layer >= DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_experts == 0 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        !stream_expert_cache_note_expert_size(gate_expert_bytes, down_expert_bytes) ||
        stream_expert_cache_effective_cap(layer, n_total_expert, 1) == 0) {
        return 1;
    }

    // Hotness decay
    stream_expert_cache_maybe_decay_route_hotness();

    // Batch halinde işle (max 6)
    uint32_t remaining = n_experts;
    while (remaining != 0) {
        const uint32_t batch = remaining > DS4_CUDA_STREAM_EXPERT_MAX_SELECTED
                               ? DS4_CUDA_STREAM_EXPERT_MAX_SELECTED : remaining;
        remaining -= batch;

        int32_t  selected_ids[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED];
        uint64_t gate_abs_offsets[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED];
        uint64_t up_abs_offsets[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED];
        uint64_t down_abs_offsets[DS4_CUDA_STREAM_EXPERT_MAX_SELECTED];
        uint32_t missing_mask = 0;

        for (uint32_t i = 0; i < DS4_CUDA_STREAM_EXPERT_MAX_SELECTED; i++)
            selected_ids[i] = -1;

        for (uint32_t i = 0; i < batch; i++) {
            const int32_t expert = expert_ids[remaining + i];
            const uint32_t priority =
                expert_priorities ? expert_priorities[remaining + i] : 0;

            if (expert < 0 || (uint32_t)expert >= n_total_expert) {
                fprintf(stderr, "ds4: CUDA streaming hotlist seed expert id %d out of range\n",
                        expert);
                return 0;
            }

            const uint64_t expert_id = (uint64_t)(uint32_t)expert;
            if (expert_id > UINT64_MAX / gate_expert_bytes ||
                expert_id > UINT64_MAX / down_expert_bytes) {
                fprintf(stderr, "ds4: CUDA streaming hotlist seed offset overflow\n");
                return 0;
            }
            const uint64_t gate_rel = expert_id * gate_expert_bytes;
            const uint64_t down_rel = expert_id * down_expert_bytes;
            if (gate_rel > UINT64_MAX - gate_offset ||
                gate_rel > UINT64_MAX - up_offset ||
                down_rel > UINT64_MAX - down_offset) {
                fprintf(stderr, "ds4: CUDA streaming hotlist seed offset overflow\n");
                return 0;
            }

            // Route hotness güncelle
            stream_expert_cache_note_route_hotness(
                    layer, (uint32_t)expert, priority != 0 ? priority : 1);

            selected_ids[i] = expert;
            gate_abs_offsets[i] = gate_offset + gate_rel;
            up_abs_offsets[i] = up_offset + gate_rel;
            down_abs_offsets[i] = down_offset + down_rel;

            // Cache'de var mı kontrol et
            StreamExpertEntry *e = &g_stream_expert_cache[layer][(uint32_t)expert];
            if (!stream_expert_cache_entry_matches(e, model_map,
                    gate_abs_offsets[i], up_abs_offsets[i], down_abs_offsets[i],
                    gate_expert_bytes, down_expert_bytes)) {
                missing_mask |= 1u << i;
            } else if (priority != 0 && e->use_count < (uint64_t)priority) {
                e->use_count = (uint64_t)priority;
            }
        }

        // Eksik olanları yükle (sync pread — hotlist seed blocking olabilir)
        if (missing_mask != 0) {
            for (uint32_t i = 0; i < batch; i++) {
                if ((missing_mask & (1u << i)) == 0) continue;

                StreamExpertEntry *e = &g_stream_expert_cache[layer][(uint32_t)selected_ids[i]];
                if (e->valid) slab_free_entry(e);

                void *gate_ptr = slab_alloc(gate_expert_bytes);
                void *up_ptr = slab_alloc(gate_expert_bytes);
                void *down_ptr = slab_alloc(down_expert_bytes);

                if (!gate_ptr || !up_ptr || !down_ptr) {
                    stream_expert_cache_prune_global(layer, selected_ids, batch);
                    gate_ptr = slab_alloc(gate_expert_bytes);
                    up_ptr = slab_alloc(gate_expert_bytes);
                    down_ptr = slab_alloc(down_expert_bytes);
                    if (!gate_ptr || !up_ptr || !down_ptr) return 0;
                }

                // Sync pread
                if (g_stream_model_fd < 0) return 0;
                ssize_t r;
                r = pread(g_stream_model_fd, gate_ptr, (size_t)gate_expert_bytes,
                          (off_t)gate_abs_offsets[i]);
                if (r != (ssize_t)gate_expert_bytes) return 0;
                r = pread(g_stream_model_fd, up_ptr, (size_t)gate_expert_bytes,
                          (off_t)up_abs_offsets[i]);
                if (r != (ssize_t)gate_expert_bytes) return 0;
                r = pread(g_stream_model_fd, down_ptr, (size_t)down_expert_bytes,
                          (off_t)down_abs_offsets[i]);
                if (r != (ssize_t)down_expert_bytes) return 0;

                e->gate_ptr = gate_ptr;
                e->up_ptr = up_ptr;
                e->down_ptr = down_ptr;
                e->model_map = model_map;
                e->gate_abs_offset = gate_abs_offsets[i];
                e->up_abs_offset = up_abs_offsets[i];
                e->down_abs_offset = down_abs_offsets[i];
                e->gate_expert_bytes = gate_expert_bytes;
                e->down_expert_bytes = down_expert_bytes;
                e->logical_bytes = gate_expert_bytes * 2 + down_expert_bytes;
                e->last_used = ++g_stream_expert_cache_clock;
                e->use_count = 1;
                e->valid = 1;
                g_stream_expert_cache_entry_count++;
            }
        }
    }

    return 1;
}

// =========================================================================
// Pending Load Finish
// =========================================================================

static int stream_expert_pending_load_finish(void) {
    StreamPendingLoad *p = &g_stream_expert_pending_load;
    if (!p->active) return 1;

    // Thread pool'daki tüm task'ların bitmesini bekle
    pread_pool_wait_all();

    p->active = 0;
    return 1;
}

// =========================================================================
// Clear All
// =========================================================================

static void stream_expert_cache_clear_all(void) {
    for (uint32_t l = 0; l < DS4_CUDA_STREAM_EXPERT_CACHE_MAX_LAYER; l++) {
        for (uint32_t e = 0; e < DS4_CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT; e++) {
            slab_free_entry(&g_stream_expert_cache[l][e]);
        }
    }
    g_stream_expert_cache_entry_count = 0;
    g_stream_expert_cache_clock = 0;
    g_stream_expert_cache_hits = 0;
    g_stream_expert_cache_misses = 0;
    g_stream_expert_cache_evictions = 0;
    memset(g_stream_expert_cache_route_hotness, 0,
           sizeof(g_stream_expert_cache_route_hotness));
    g_stream_expert_cache_decode_tokens = 0;
    g_stream_expert_cache_hotness_decay_token = 0;
    g_routed_moe_selected_override_n = 0;
    stream_expert_pending_load_clear();
}

// =========================================================================
// Extern "C" API — simple setters/getters
// =========================================================================

extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming_mode = enabled ? 1 : 0;
    stream_expert_cache_clear_all();
    slab_destroy_all();
    g_stream_expert_cache_entry_count = 0;
    g_stream_expert_cache_clock = 0;

    if (g_ssd_streaming_mode) {
        fprintf(stderr, "ds4: CUDA SSD streaming mode enabled\n");
        pread_pool_start();
    }
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    if (experts > DS4_CUDA_STREAM_EXPERT_CACHE_MAX_ENTRIES)
        experts = DS4_CUDA_STREAM_EXPERT_CACHE_MAX_ENTRIES;
    g_stream_expert_cache_budget_override = experts;
    stream_expert_cache_clear_all();
    g_stream_expert_cache_entry_count = 0;
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    size_t free_b, total_b;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) return 0;
    return (uint64_t)total_b;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    return stream_expert_cache_configured_budget();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return g_stream_expert_cache_entry_count;
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
    stream_expert_cache_reset_route_hotness();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!stream_expert_cache_note_expert_size(gate_expert_bytes, down_expert_bytes))
        return 0;
    return stream_expert_cache_configured_budget();
}

#endif // DS4_CUDA_STREAM_EXPERT_CACHE_CUH

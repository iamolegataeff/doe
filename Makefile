CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS  = -lm -lpthread

.PHONY: all test clean run run-int8 run-smollm360 run-f16 quantize-q4

all: doe_field

doe_field: doe.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

# macOS Accelerate
blas: doe.c
	$(CC) $(CFLAGS) -DUSE_BLAS -DACCELERATE $< $(LDFLAGS) -framework Accelerate -o doe_field

# OpenBLAS (Linux)
openblas: doe.c
	$(CC) $(CFLAGS) -DUSE_BLAS $< $(LDFLAGS) -lopenblas -o doe_field

# cuBLAS (NVIDIA GPU)
cuda: doe.c
	$(CC) $(CFLAGS) -DUSE_CUBLAS $< $(LDFLAGS) -lcublas -lcudart -o doe_field

test: tests/test_doe.c doe.c
	$(CC) $(CFLAGS) tests/test_doe.c $(LDFLAGS) -o tests/test_doe
	./tests/test_doe

# --- personality weights (HF ataeff/janus) + Q4_0 quantization ---
HF_BASE = https://huggingface.co/ataeff/janus/resolve/main/DoE
WEIGHTS = weights

$(WEIGHTS)/doe_smollm360_lora_1000.gguf:
	@mkdir -p $(WEIGHTS)
	curl -fL -o $@ $(HF_BASE)/doe_smollm360_lora_1000.gguf

$(WEIGHTS)/doe_qwen15b_lora_1000.gguf:
	@mkdir -p $(WEIGHTS)
	curl -fL -o $@ $(HF_BASE)/doe_qwen15b_lora_1000.gguf

# Q4_0 — ~3.2x smaller, the parliament survives the quant, and it lights up the int8
# fast path. llama.cpp is used only as a GGUF converter here.
$(WEIGHTS)/doe_qwen15b_q4_0.gguf: $(WEIGHTS)/doe_qwen15b_lora_1000.gguf
	llama-quantize $< $@ Q4_0

quantize-q4: $(WEIGHTS)/doe_qwen15b_q4_0.gguf

# default run: Qwen2.5-1.5B Q4_0 — small, fits on 8 GB, parliament intact
run: doe_field $(WEIGHTS)/doe_qwen15b_q4_0.gguf
	./doe_field --model $(WEIGHTS)/doe_qwen15b_q4_0.gguf

# Q4_0 + int8 dynamic-activation-quant fast path (NEON SDOT, approximate)
run-int8: doe_field $(WEIGHTS)/doe_qwen15b_q4_0.gguf
	DOE_INT8=1 ./doe_field --model $(WEIGHTS)/doe_qwen15b_q4_0.gguf

run-smollm360: doe_field $(WEIGHTS)/doe_smollm360_lora_1000.gguf
	./doe_field --model $(WEIGHTS)/doe_smollm360_lora_1000.gguf

run-f16: doe_field $(WEIGHTS)/doe_qwen15b_lora_1000.gguf
	./doe_field --model $(WEIGHTS)/doe_qwen15b_lora_1000.gguf

clean:
	rm -f doe_field tests/test_doe

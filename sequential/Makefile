# pdl-bench — SEQUENTIAL version (plain C: no MPI, no OpenMP, no CUDA)
# SPDX-License-Identifier: MIT
#
#   make                 # builds ./seq_train with $(CC) (gcc/clang)
#   make run             # 2 epochs of LeNet on MNIST (needs ../data/mnist-*.pdlds)
#   make run-synthetic   # no dataset needed
#   make clean

CC      ?= gcc
CFLAGS  ?= -O3 -march=native
CFLAGS  += -std=gnu11 -Wall -Wextra -Wno-unused-parameter
CPPFLAGS += -I../include -I../src/backend -DPDL_BUILD_TYPE='"Release"'
LDLIBS  += -lm

# Shared CNN core only — the parallel parts (src/comm, src/train, CUDA) are not used.
SRCS = seq_train.c \
       ../src/core/common.c ../src/core/timer.c ../src/core/rng.c \
       ../src/core/json.c ../src/core/cli.c \
       ../src/backend/backend.c ../src/backend/cpu_kernels.c \
       ../src/nn/model.c ../src/data/dataset.c
OBJS = $(patsubst %.c,obj/%.o,$(notdir $(SRCS)))
VPATH = ../src/core ../src/backend ../src/nn ../src/data .

seq_train: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

obj/%.o: %.c | obj
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

run: seq_train
	./seq_train --epochs 2 --output seq_mnist.json

run-synthetic: seq_train
	./seq_train --train-data synthetic:4096:1:28:28:10 --test-data synthetic:1024:1:28:28:10 \
	            --epochs 2 --output seq_synthetic.json

clean:
	rm -rf obj seq_train *.json

.PHONY: run run-synthetic clean

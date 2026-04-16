CLANG = clang
CFLAGS = -O3 -ffast-math -fopenmp -Wall -Wextra
all:mb_tiny_test.exe

mb_tiny_test.exe:mb_tiny.c mb_tiny_test.c mb_tiny.pth.c
	$(CLANG) -target x86_64-w64-mingw32 $(CFLAGS) -march=native $^ -o $@

mb_tiny_test.musl.static.amd64.elf:mb_tiny.c mb_tiny_test.c mb_tiny.pth.c
	$(CLANG) -target x86_64-linux-musl -static $(CFLAGS) -march=native $^ -o $@

mb_tiny_test.android.aarch64.elf:mb_tiny.c mb_tiny_test.c mb_tiny.pth.c
	$(CLANG) -target aarch64-linux-android $(CFLAGS) -pie $^ -o $@

mb_tiny.wasm:mb_tiny.c mb_tiny.pth.c wasm.c
	$(CLANG) --target=wasm32 -static -O3 -ffast-math -nostdlib -Wl,--no-entry $^ -o $@

release:mb_tiny_test.exe mb_tiny_test.musl.static.amd64.elf mb_tiny_test.android.aarch64.elf mb_tiny.wasm
	llvm-strip $^
	python wasm2html.py

clean:
	rm -f *.o *.exe

run: mb_tiny_test.exe
	./$<

.PHONY: all clean run

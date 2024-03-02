## Cross Compile

Install cross compile tools:

``` bash
$ sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

Or install the binaries from: https://releases.linaro.org/components/toolchain/binaries/latest-7


## Build

``` bash
$ cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux-gnu.toolchain.cmake
$ cmake --build build
```

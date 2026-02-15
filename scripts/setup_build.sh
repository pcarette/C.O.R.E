#!/bin/sh

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CUDACXX=/usr/bin/clang++

if [ -d build ]; then
  echo "build dir already exists"
  exit 1
fi

mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release

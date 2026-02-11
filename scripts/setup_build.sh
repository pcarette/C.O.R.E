#!/bin/sh

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CUDACXX=/usr/bin/clang++

if [ -d cmake-build-release ]; then
  echo "build dir already exists"
  exit 1
fi

mkdir cmake-build-release
cd cmake-build-release
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release

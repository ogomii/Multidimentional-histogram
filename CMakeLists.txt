cmake_minimum_required(VERSION 3.12)
project(my_module)

set(CMAKE_PREFIX_PATH /home/skili/.local/lib/python3.10/site-packages/pybind11)

find_package(pybind11 REQUIRED)

# Ajouter le chemin vers le répertoire contenant le fichier d'en-tête généré
include_directories(${CMAKE_PREFIX_PATH}/include)

pybind11_add_module(my_module src_cpp/Interface.cpp src_python/baseline.py)


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

py::tuple calculate_histogram_cpp(py::array_t<double> data, py::list bins) {
    // Convertir le tableau NumPy en vecteur C++
    auto buf = data.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<double> cpp_data(ptr, ptr + buf.size);

    // Importer explicitement le module Python avec le chemin complet
    py::object baseline_module = py::module::import("src_python.baseline");

    // Accéder à la fonction par son nom
    py::object histogram_func = baseline_module.attr("histogram");

    // Créer un tableau bidimensionnel (N, D) à partir du vecteur C++
    int N = buf.size;
    int D = 1;  // Vous devez définir D en fonction de votre logique
    py::array_t<double> np_data({N, D}, cpp_data.data());

    // Appeler la fonction Python avec le tableau bidimensionnel et bins
    py::tuple result = histogram_func(np_data, bins);

    return result;
}

// Module Python
PYBIND11_MODULE(my_module, m) {
    m.def("calculate_histogram", &calculate_histogram_cpp, "Calculate histogram using NumPy");
}
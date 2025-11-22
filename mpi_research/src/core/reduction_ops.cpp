#include "reduction_ops.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <complex>
#include <type_traits>

namespace TopologyAwareResearch {

// Global custom op manager
CustomOpManager g_custom_op_manager;

// Helper function to get MPI type size
int get_mpi_type_size(MPI_Datatype datatype) {
    int size;
    MPI_Type_size(datatype, &size);
    return size;
}

// Helper function to compare MPI operations
bool is_mpi_op_equal(MPI_Op op1, MPI_Op op2) {
    // Compare MPI operations by converting to integers
    // This is a workaround since we can't directly compare MPI_Op
    int commutative1, commutative2;
    MPI_Op_commutative(op1, &commutative1);
    MPI_Op_commutative(op2, &commutative2);

    // If both are commutative and we're comparing with common ops,
    // we can make reasonable assumptions
    return (op1 == op2); // Direct comparison might work with some MPI implementations
}

// Template function for reduction operations
template<typename T>
void reduce_elements(T* dest, const T* src, int count, MPI_Op op) {
    // Use direct comparison with MPI predefined operations
    if (op == MPI_SUM) {
        for (int i = 0; i < count; ++i) {
            dest[i] += src[i];
        }
    }
    else if (op == MPI_PROD) {
        for (int i = 0; i < count; ++i) {
            dest[i] *= src[i];
        }
    }
    else if (op == MPI_MAX) {
        for (int i = 0; i < count; ++i) {
            if (src[i] > dest[i]) {
                dest[i] = src[i];
            }
        }
    }
    else if (op == MPI_MIN) {
        for (int i = 0; i < count; ++i) {
            if (src[i] < dest[i]) {
                dest[i] = src[i];
            }
        }
    }
    else if (op == MPI_LAND) {
        for (int i = 0; i < count; ++i) {
            dest[i] = dest[i] && src[i];
        }
    }
    else if (op == MPI_BAND) {
        // Only for integral types
        if constexpr (std::is_integral_v<T>) {
            for (int i = 0; i < count; ++i) {
                dest[i] = dest[i] & src[i];
            }
        }
    }
    else if (op == MPI_LOR) {
        for (int i = 0; i < count; ++i) {
            dest[i] = dest[i] || src[i];
        }
    }
    else if (op == MPI_BOR) {
        // Only for integral types
        if constexpr (std::is_integral_v<T>) {
            for (int i = 0; i < count; ++i) {
                dest[i] = dest[i] | src[i];
            }
        }
    }
    else if (op == MPI_LXOR) {
        for (int i = 0; i < count; ++i) {
            dest[i] = (dest[i] && !src[i]) || (!dest[i] && src[i]);
        }
    }
    else if (op == MPI_BXOR) {
        // Only for integral types
        if constexpr (std::is_integral_v<T>) {
            for (int i = 0; i < count; ++i) {
                dest[i] = dest[i] ^ src[i];
            }
        }
    }
    else if (op == MPI_MAXLOC) {
        // Handle MAXLOC operation
        struct MaxLocType { T value; int index; };
        MaxLocType* dest_pairs = reinterpret_cast<MaxLocType*>(dest);
        const MaxLocType* src_pairs = reinterpret_cast<const MaxLocType*>(src);

        for (int i = 0; i < count; ++i) {
            if (src_pairs[i].value > dest_pairs[i].value ||
               (src_pairs[i].value == dest_pairs[i].value && src_pairs[i].index < dest_pairs[i].index)) {
                dest_pairs[i] = src_pairs[i];
            }
        }
    }
    else if (op == MPI_MINLOC) {
        // Handle MINLOC operation
        struct MinLocType { T value; int index; };
        MinLocType* dest_pairs = reinterpret_cast<MinLocType*>(dest);
        const MinLocType* src_pairs = reinterpret_cast<const MinLocType*>(src);

        for (int i = 0; i < count; ++i) {
            if (src_pairs[i].value < dest_pairs[i].value ||
               (src_pairs[i].value == dest_pairs[i].value && src_pairs[i].index < dest_pairs[i].index)) {
                dest_pairs[i] = src_pairs[i];
            }
        }
    }
    else if (op == MPI_REPLACE) {
        std::memcpy(dest, src, count * sizeof(T));
    }
    else {
        // Handle custom operations or fallback to SUM
        for (int i = 0; i < count; ++i) {
            dest[i] += src[i];
        }
    }
}

// Helper function to convert MPI_Datatype to comparable value
int datatype_to_int(MPI_Datatype datatype) {
    // Compare with known MPI datatypes
    if (datatype == MPI_CHAR) return 1;
    if (datatype == MPI_SHORT) return 2;
    if (datatype == MPI_INT) return 3;
    if (datatype == MPI_LONG) return 4;
    if (datatype == MPI_UNSIGNED_CHAR) return 5;
    if (datatype == MPI_UNSIGNED_SHORT) return 6;
    if (datatype == MPI_UNSIGNED) return 7;
    if (datatype == MPI_UNSIGNED_LONG) return 8;
    if (datatype == MPI_FLOAT) return 9;
    if (datatype == MPI_DOUBLE) return 10;
    if (datatype == MPI_LONG_DOUBLE) return 11;
    if (datatype == MPI_BYTE) return 12;
    if (datatype == MPI_C_COMPLEX) return 13;
    if (datatype == MPI_C_DOUBLE_COMPLEX) return 14;
    return 0; // Unknown
}

// Main reduction function
void reduce_segments(void* dest, void* src, int start, int count,
                    MPI_Datatype datatype, MPI_Op op) {
    // Handle MPI_IN_PLACE
    if (src == MPI_IN_PLACE) {
        return;
    }

    // Check for custom operations first
    if (g_custom_op_manager.has_custom_op(op)) {
        g_custom_op_manager.apply_custom_op(dest, src, start, count, datatype, op);
        return;
    }

    // Use if-else instead of switch for MPI_Datatype
    int dt = datatype_to_int(datatype);

    if (dt == 1) { // MPI_CHAR
        char* dest_c = static_cast<char*>(dest) + start;
        const char* src_c = static_cast<const char*>(src);
        reduce_elements(dest_c, src_c, count, op);
    }
    else if (dt == 2) { // MPI_SHORT
        short* dest_s = static_cast<short*>(dest) + start;
        const short* src_s = static_cast<const short*>(src);
        reduce_elements(dest_s, src_s, count, op);
    }
    else if (dt == 3) { // MPI_INT
        int* dest_i = static_cast<int*>(dest) + start;
        const int* src_i = static_cast<const int*>(src);
        reduce_elements(dest_i, src_i, count, op);
    }
    else if (dt == 4) { // MPI_LONG
        long* dest_l = static_cast<long*>(dest) + start;
        const long* src_l = static_cast<const long*>(src);
        reduce_elements(dest_l, src_l, count, op);
    }
    else if (dt == 5) { // MPI_UNSIGNED_CHAR
        unsigned char* dest_uc = static_cast<unsigned char*>(dest) + start;
        const unsigned char* src_uc = static_cast<const unsigned char*>(src);
        reduce_elements(dest_uc, src_uc, count, op);
    }
    else if (dt == 6) { // MPI_UNSIGNED_SHORT
        unsigned short* dest_us = static_cast<unsigned short*>(dest) + start;
        const unsigned short* src_us = static_cast<const unsigned short*>(src);
        reduce_elements(dest_us, src_us, count, op);
    }
    else if (dt == 7) { // MPI_UNSIGNED
        unsigned* dest_u = static_cast<unsigned*>(dest) + start;
        const unsigned* src_u = static_cast<const unsigned*>(src);
        reduce_elements(dest_u, src_u, count, op);
    }
    else if (dt == 8) { // MPI_UNSIGNED_LONG
        unsigned long* dest_ul = static_cast<unsigned long*>(dest) + start;
        const unsigned long* src_ul = static_cast<const unsigned long*>(src);
        reduce_elements(dest_ul, src_ul, count, op);
    }
    else if (dt == 9) { // MPI_FLOAT
        float* dest_f = static_cast<float*>(dest) + start;
        const float* src_f = static_cast<const float*>(src);
        reduce_elements(dest_f, src_f, count, op);
    }
    else if (dt == 10) { // MPI_DOUBLE
        double* dest_d = static_cast<double*>(dest) + start;
        const double* src_d = static_cast<const double*>(src);
        reduce_elements(dest_d, src_d, count, op);
    }
    else if (dt == 11) { // MPI_LONG_DOUBLE
        long double* dest_ld = static_cast<long double*>(dest) + start;
        const long double* src_ld = static_cast<const long double*>(src);
        reduce_elements(dest_ld, src_ld, count, op);
    }
    else if (dt == 12) { // MPI_BYTE
        char* dest_b = static_cast<char*>(dest) + start;
        const char* src_b = static_cast<const char*>(src);

        // Handle byte operations
        if (op == MPI_BAND || op == MPI_BOR || op == MPI_BXOR) {
            for (int i = 0; i < count; ++i) {
                if (op == MPI_BAND) dest_b[i] &= src_b[i];
                else if (op == MPI_BOR) dest_b[i] |= src_b[i];
                else if (op == MPI_BXOR) dest_b[i] ^= src_b[i];
            }
        } else if (op == MPI_REPLACE) {
            std::memcpy(dest_b, src_b, count);
        }
    }
    else {
        // Handle unknown datatype with memcpy for REPLACE operation
        char* dest_unknown = static_cast<char*>(dest) + start * get_mpi_type_size(datatype);
        const char* src_unknown = static_cast<const char*>(src);

        if (op == MPI_REPLACE) {
            std::memcpy(dest_unknown, src_unknown, count * get_mpi_type_size(datatype));
        }
    }
}

// Check if operation is supported for datatype
bool is_operation_supported(MPI_Datatype datatype, MPI_Op op) {
    if (op == MPI_LAND || op == MPI_LOR || op == MPI_LXOR) {
        return (datatype == MPI_CHAR || datatype == MPI_SHORT || datatype == MPI_INT ||
                datatype == MPI_LONG || datatype == MPI_UNSIGNED_CHAR ||
                datatype == MPI_UNSIGNED_SHORT || datatype == MPI_UNSIGNED ||
                datatype == MPI_UNSIGNED_LONG);
    }

    if (op == MPI_BAND || op == MPI_BOR || op == MPI_BXOR) {
        return (datatype == MPI_CHAR || datatype == MPI_SHORT || datatype == MPI_INT ||
                datatype == MPI_LONG || datatype == MPI_UNSIGNED_CHAR ||
                datatype == MPI_UNSIGNED_SHORT || datatype == MPI_UNSIGNED ||
                datatype == MPI_UNSIGNED_LONG || datatype == MPI_BYTE);
    }

    if (op == MPI_MAXLOC || op == MPI_MINLOC) {
        return (datatype == MPI_FLOAT || datatype == MPI_DOUBLE || datatype == MPI_INT ||
                datatype == MPI_LONG);
    }

    return true;
}

// Check if SIMD optimization can be used
bool is_simd_supported(MPI_Datatype datatype, MPI_Op op) {
    if (op != MPI_SUM && op != MPI_PROD && op != MPI_MAX && op != MPI_MIN) {
        return false;
    }

    return (datatype == MPI_FLOAT || datatype == MPI_DOUBLE ||
            datatype == MPI_INT || datatype == MPI_LONG);
}

// SIMD-optimized reduction (placeholder)
PerformanceMetrics simd_reduce_segments(void* dest, void* src, int start, int count,
                                       MPI_Datatype datatype, MPI_Op op) {
    PerformanceMetrics metrics;
    auto start_time = MPI_Wtime();

    reduce_segments(dest, src, start, count, datatype, op);

    auto end_time = MPI_Wtime();
    metrics.execution_time = end_time - start_time;
    metrics.computation_time = metrics.execution_time;

    return metrics;
}

// Optimized reduction with error checking
PerformanceMetrics optimized_reduce_segments(void* dest, void* src, int start, int count,
                                           MPI_Datatype datatype, MPI_Op op,
                                           const NetworkCharacteristics& network_config) {
    PerformanceMetrics metrics;
    auto start_time = MPI_Wtime();

    if (dest == nullptr || (src == nullptr && src != MPI_IN_PLACE)) {
        std::cerr << "Error: Invalid buffer pointers in reduce_segments" << std::endl;
        metrics.execution_time = 0.0;
        return metrics;
    }

    if (count <= 0) {
        metrics.execution_time = 0.0;
        return metrics;
    }

    if (!is_operation_supported(datatype, op)) {
        std::cerr << "Warning: Operation " << op << " may not be well-defined for datatype "
                  << datatype << ". Using MPI_SUM as fallback." << std::endl;
        op = MPI_SUM;
    }

    if (count > 1000 && is_simd_supported(datatype, op)) {
        metrics = simd_reduce_segments(dest, src, start, count, datatype, op);
    } else {
        reduce_segments(dest, src, start, count, datatype, op);
    }

    auto end_time = MPI_Wtime();
    metrics.execution_time = end_time - start_time;
    metrics.computation_time = metrics.execution_time;
    // Remove the line causing error - bytes_processed doesn't exist
    // metrics.bytes_processed = count * get_mpi_type_size(datatype);

    return metrics;
}

// CustomOpManager implementation
void CustomOpManager::register_custom_op(MPI_Op op, MPI_User_function* function,
                                       void* extra_data, bool commutative) {
    custom_ops_[op] = {function, extra_data, commutative};
}

bool CustomOpManager::has_custom_op(MPI_Op op) const {
    return custom_ops_.find(op) != custom_ops_.end();
}

void CustomOpManager::apply_custom_op(void* dest, void* src, int start, int count,
                                    MPI_Datatype datatype, MPI_Op op) {
    auto it = custom_ops_.find(op);
    if (it != custom_ops_.end()) {
        CustomReduceOp& custom_op = it->second;
        char* dest_ptr = static_cast<char*>(dest) + start * get_mpi_type_size(datatype);
        char* src_ptr = static_cast<char*>(src);

        for (int i = 0; i < count; ++i) {
            custom_op.function(dest_ptr + i * get_mpi_type_size(datatype),
                              src_ptr + i * get_mpi_type_size(datatype),
                              &count, &datatype);
        }
    }
}

} // namespace TopologyAwareResearch
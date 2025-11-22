#ifndef REDUCTION_OPS_H
#define REDUCTION_OPS_H

#include <mpi.h>
#include <map>
#include <functional>
#include "collective_optimizer.h"

namespace TopologyAwareResearch {

// Helper function declarations
int get_mpi_type_size(MPI_Datatype datatype);
bool is_operation_supported(MPI_Datatype datatype, MPI_Op op);
bool is_simd_supported(MPI_Datatype datatype, MPI_Op op);

// Main reduction functions
void reduce_segments(void* dest, void* src, int start, int count,
                    MPI_Datatype datatype, MPI_Op op);

PerformanceMetrics optimized_reduce_segments(void* dest, void* src, int start, int count,
                                           MPI_Datatype datatype, MPI_Op op,
                                           const NetworkCharacteristics& network_config);

// Custom operation support
struct CustomReduceOp {
    MPI_User_function* function;
    void* extra_data;
    bool commutative;
};

class CustomOpManager {
private:
    std::map<MPI_Op, CustomReduceOp> custom_ops_;

public:
    void register_custom_op(MPI_Op op, MPI_User_function* function,
                           void* extra_data, bool commutative);
    bool has_custom_op(MPI_Op op) const;
    void apply_custom_op(void* dest, void* src, int start, int count,
                        MPI_Datatype datatype, MPI_Op op);
};

// External declaration for global custom op manager
extern CustomOpManager g_custom_op_manager;

} // namespace TopologyAwareResearch

#endif // REDUCTION_OPS_H
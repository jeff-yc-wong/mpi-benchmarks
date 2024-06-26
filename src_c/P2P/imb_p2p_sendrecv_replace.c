/****************************************************************************
*                                                                           *
* Copyright (C) 2020 Intel Corporation                                      *
*                                                                           *
*****************************************************************************

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ***************************************************************************/

#include "imb_p2p.h"

void imb_p2p_sendrecv_replace() {
    char *buffer;
    size_t msg_size_index;
    int nranks = imb_p2p_config.nranks;
    if (nranks < 2) {
        if (imb_p2p_config.rank == 0) {
            fprintf(unit, "\n");
            fprintf(unit, "# !! Benchmark %s is invalid for %d process !!\n", IMB_P2P_SENDRECV_REPLACE, nranks);
            fflush(unit);
        }
        return;
    }
    if (imb_p2p_config.rank == 0) {
        imb_p2p_print_benchmark_header(IMB_P2P_SENDRECV_REPLACE);
        fprintf(unit, " %12s %12s %12s %12s %12s\n", "#bytes", "#repetitions", "t[usec]", "Mbytes/sec", "Msg/sec");
        fflush(unit);
    }
    buffer = (char *)SMPI_SHARED_MALLOC(imb_p2p_config.messages.max_size);
    memset(buffer, imb_p2p_config.rank, imb_p2p_config.messages.max_size);
    for (msg_size_index = 0; msg_size_index < imb_p2p_config.messages.length; msg_size_index++) {
        size_t size = imb_p2p_config.messages.array[msg_size_index];
        size_t iteration, number_of_iterations, number_of_warm_up_iterations;
        double time;
        int dst = (imb_p2p_config.rank + 1) % nranks;
        int src = (imb_p2p_config.rank + nranks - 1) % nranks;
        get_iters(size, &number_of_iterations, &number_of_warm_up_iterations);
        imb_p2p_pause();
        imb_p2p_barrier(MPI_COMM_WORLD);
        for (iteration = 0; iteration < number_of_warm_up_iterations; iteration++) {
            touch_send_buff(size, buffer);
            MPI_Sendrecv_replace(buffer, size, MPI_BYTE,
                                 dst, 0, src, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        touch_recv_buff(size, buffer);
        time = MPI_Wtime();
        for (iteration = 0; iteration < number_of_iterations; iteration++) {
            touch_send_buff(size, buffer);
            MPI_Sendrecv_replace(buffer, size, MPI_BYTE,
                                 dst, 0, src, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        touch_recv_buff(size, buffer);
        time = MPI_Wtime() - time;
        imb_p2p_pause();
        imb_p2p_barrier(MPI_COMM_WORLD);
        if (imb_p2p_config.rank) {
            MPI_Send(&time, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        } else {
            const double microseconds = 1000000.0;
            double bandwidth, message_rate, latency;
            double max_time = time;
            double aggregate_time = time;
            int source_rank;
            for (source_rank = 1; source_rank < nranks; source_rank++) {
                double t = 0;
                MPI_Recv(&t, 1, MPI_DOUBLE, source_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                aggregate_time += t;
                if (max_time < t) {
                    max_time = t;
                }
            }
            bandwidth = ((number_of_iterations * nranks * size) / (microseconds * max_time));
            message_rate = ((number_of_iterations * nranks) / max_time);
            latency = (microseconds * aggregate_time) / (number_of_iterations * nranks);
            fprintf(unit, " %12" PRIu64 " %12" PRIu64 " %12.2f %12.2f %12.0f\n", size, number_of_iterations, latency, bandwidth, message_rate);
            fflush(unit);
        }
    }
}

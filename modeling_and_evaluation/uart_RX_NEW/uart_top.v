`timescale 1ns / 1ps

module uart_top
    #(
        parameter   DBITS          = 8,
                    SB_TICK        = 16,
                    BR_LIMIT       = 68,
                    BR_BITS        = 7,
                    FIFO_ADDR_BITS = 4
    )
    (
        input  clk,                           // 125 MHz FPGA clock (renamed to avoid confusion)
        input  reset,
        input  rx,
        input  read_from_fifo,
        output [DBITS-1:0] data_out,
        output fifo_empty,
        output fifo_full
    );

    wire sample_tick;
    wire data_ready;
    wire [DBITS-1:0] rx_data_out;

    baud_rate_generator
        #(.N(BR_BITS), .M(BR_LIMIT))
        baud_gen_unit
        (
            .clk_100MHz(clk),
            .reset(reset),
            .tick(sample_tick)
        );

    uart_receiver
        #(.DBITS(DBITS), .SB_TICK(SB_TICK))
        uart_rx_unit
        (
            .clk_100MHz(clk),
            .reset(reset),
            .rx(rx),
            .sample_tick(sample_tick),
            .data_ready(data_ready),
            .data_out(rx_data_out)
        );

    sync_fifo
        #(.DEPTH(1 << FIFO_ADDR_BITS), .WIDTH(DBITS))
        fifo_unit
        (
            .clk(clk),
            .rst_n(reset),
            .wr_en(data_ready),
            .rd_en(read_from_fifo),
            .wr_data(rx_data_out),
            .rd_data(data_out),
            .full(fifo_full),
            .empty(fifo_empty)
        );

endmodule

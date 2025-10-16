`timescale 1ns / 1ps


module tile_load_top #(
    parameter DATA_WIDTH = 12,
    parameter ADDR_BITS = 14,
    parameter DEPTH = 16*16

) (
    input clk,
    input rst,
    input start,
    output done
);

wire [DATA_WIDTH-1:0] data_out_mem;
wire [DATA_WIDTH-1:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4, data_out_5, data_out_6, data_out_7,
                      data_out_8, data_out_9, data_out_10, data_out_11, data_out_12, data_out_13, data_out_14, data_out_15;
wire [ADDR_BITS-1:0] current_addr;
wire [ADDR_BITS-1:0] waddr_systolic;
wire we_sys;
wire fifo_full, fifo_empty;
wire [DATA_WIDTH-1:0] pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8, pe9, pe10, pe11, pe12, pe13, pe14, pe15, pe16;
wire [DATA_WIDTH-1:0] data_row_in;
wire [DATA_WIDTH-1:0] data_row_out;
wire [ADDR_BITS-1:0] row_addr;
wire wr_en_row;



main_mem #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(ADDR_BITS),
    .DEPTH(256)
) u_main_mem (
    .clk(clk),
    .rst(rst),
    .we(1'b0),
    .data_in(),
    .waddr(),
    .raddr(current_addr),
    .data_out(data_out_mem)
);

bram_burst #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_WIDTH(ADDR_BITS),
    .NUM_PORTS(16),
    .MATRIX_WIDTH(16)
) u_bram_burst (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_in(data_out_mem),
    .base_addr(0), // Starting address for reading from main memory
    .base_addr_row(0), // Starting address for writing rows
    .waddr_row(waddr_data), // Address for writing rows
    .data_row_in(data_row_in), // Data input for writing rows
    .data_row_out(data_row_out), // Data output for writing rows
    .row_addr(row_addr), // Address for row memory
    .wr_en_row(wr_en_row), // Write enable for row memory
    .we(we_sys),
    .data_out_0(data_out_0),
    .data_out_1(data_out_1),
    .data_out_2(data_out_2),
    .data_out_3(data_out_3),
    .data_out_4(data_out_4),
    .data_out_5(data_out_5),
    .data_out_6(data_out_6),
    .data_out_7(data_out_7),
    .data_out_8(data_out_8),
    .data_out_9(data_out_9),
    .data_out_10(data_out_10),
    .data_out_11(data_out_11),
    .data_out_12(data_out_12),
    .data_out_13(data_out_13),
    .data_out_14(data_out_14),
    .data_out_15(data_out_15),
    .current_addr(current_addr),
    .done(done),
    .waddr(waddr_sys)
);

top_16_by_1 #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(DATA_WIDTH),
    .FIFO_DEPTH(16)
) u_top_16_by_1 (
    .clk(clk),
    .rst(rst),
    .wr_en(we_sys),
    .wr_en_data(wr_en_row),
    .data_r1(data_row_out), // row data rows of the matrix 
    .weight_c1(data_out_0),.weight_c2(data_out_1),.weight_c3(data_out_2),.weight_c4(data_out_3),.weight_c5(data_out_4),.weight_c6(data_out_5),.weight_c7(data_out_6),.weight_c8(data_out_7),.weight_c9(data_out_8),.weight_c10(data_out_9),.weight_c11(data_out_10),.weight_c12(data_out_11),.weight_c13(data_out_12),.weight_c14(data_out_13),.weight_c15(data_out_14),.weight_c16(data_out_15),// column of the matrix 
    .pe1(pe1),.pe2(pe2),.pe3(pe3),.pe4(pe4),.pe5(pe5),.pe6(pe6),.pe7(pe7),.pe8(pe8),.pe9(pe9),.pe10(pe10),.pe11(pe11),.pe12(pe12),.pe13(pe13),.pe14(pe14),.pe15(pe15),.pe16(pe16),// processing element outputs
    .fifo_full(fifo_full),
    .fifo_empty(fifo_empty)
);

data_mem #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(ADDR_BITS),
    .DEPTH(DEPTH)
) u_data_mem (
    .clk(clk),
    .rst(rst),
    .we(0),
    .data_in(0), // Example: writing pe1 output to memory
    .waddr(0),
    .raddr(row_addr),
    .data_out(data_row_in)
);

endmodule

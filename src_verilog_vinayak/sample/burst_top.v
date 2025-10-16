module burst_top #(
    parameter DATA_WIDTH = 12,
    parameter ADDR_WIDTH = 14

) (
    input clk,
    input rst,
    input start,
    output done
);
wire [4-1:0] write_addr;
wire we;
wire [DATA_WIDTH-1:0] data_out_0, data_out_1, data_out_2, data_out_3, data_out_4, data_out_5, data_out_6, data_out_7,
                     data_out_8, data_out_9, data_out_10, data_out_11, data_out_12, data_out_13, data_out_14, data_out_15;

wire [ADDR_WIDTH-1:0] current_addr;
wire [DATA_WIDTH-1:0] data_out_main_mem;
wire [DATA_WIDTH-1:0] data_out_lutram, data_out_lutram_1, data_out_lutram_2, data_out_lutram_3, data_out_lutram_4,
                     data_out_lutram_5, data_out_lutram_6, data_out_lutram_7, data_out_lutram_8, data_out_lutram_9,
                     data_out_lutram_10, data_out_lutram_11, data_out_lutram_12, data_out_lutram_13, data_out_lutram_14,
                     data_out_lutram_15;
wire [DATA_WIDTH-1:0] data_in; // Example data input from LUTRAM


bram_burst #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .NUM_PORTS(16)
) burst_inst (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_in(data_out_main_mem), // Not used in this example
    .base_addr(0),    // Base address for burst operation
    .we(we),
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
    .waddr(write_addr) // Not used in this example
);

lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_0 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_0), // Example data input
    .data_out(data_out_lutram) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_1 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_1), // Example data input
    .data_out(data_out_lutram_1) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_2 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_2), // Example data input
    .data_out(data_out_lutram_2) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_3 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_3), // Example data input
    .data_out(data_out_lutram_3) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_4 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_4), // Example data input
    .data_out(data_out_lutram_4) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_5 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_5), // Example data input
    .data_out(data_out_lutram_5) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_6 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_6), // Example data input
    .data_out(data_out_lutram_6) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_7 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_7), // Example data input
    .data_out(data_out_lutram_7) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_8 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_8), // Example data input
    .data_out(data_out_lutram_8) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_9 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_9), // Example data input
    .data_out(data_out_lutram_9) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_10 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_10), // Example data input
    .data_out(data_out_lutram_10) // Example data output
);

lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_11 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_11), // Example data input
    .data_out(data_out_lutram_11) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_12 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_12), // Example data input
    .data_out(data_out_lutram_12) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_13 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_13), // Example data input
    .data_out(data_out_lutram_13) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_14 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_14), // Example data input
    .data_out(data_out_lutram_14) // Example data output
);
lutram_simple #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(4),
    .DEPTH(16)
) lutram_inst_15 (
    .clk(clk),
    .we(we),
    .addr(write_addr), // Address to write/read
    .data_in(data_out_15), // Example data input
    .data_out(data_out_lutram_15) // Example data output
);
main_mem #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_BITS(14),
    .DEPTH(96*96)
) main_mem_inst (
    .clk(clk),
    .rst(rst),
    .we(0),
    .data_in(data_in), // Example data input from LUTRAM
    .waddr(0), // Address to write/read
    .raddr(current_addr), // Address to write/read
    .data_out(data_out_main_mem) // Example data output
);
endmodule

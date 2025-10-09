module main (
    input wire clk,
    input wire rst,
    input wire wr_en,
    input wire [11:0] data_r1, // row data rows of the matrix 
    input wire [11:0] weight_c1,weight_c2,weight_c3,weight_c4 ,// column of the matrix 
    output wire [11:0] pe11,pe12,pe13,pe14,// processing element outputs
    output wire fifo_full,
    output wire fifo_empty,
    output wire fifo_valid
);
    // Instantiate the 4x1 top module
 top_4_by_1 #(
        .DATA_WIDTH(12),
        .OUTPUT_WIDTH(12),
        .FIFO_DEPTH(16)
    ) dut (
        .clk(clk),
        .rst(rst),
        .wr_en(wr_en),
        .data_r1(data_r1),
        .weight_c1(weight_c1),.weight_c2(weight_c2), .weight_c3(weight_c3), .weight_c4(weight_c4),
        .pe11(pe11),   .pe12(pe12),   .pe13(pe13),   .pe14(pe14),
        .fifo_full(fifo_full),
        .fifo_empty(fifo_empty),
        .fifo_valid(fifo_valid)
    );
    
endmodule
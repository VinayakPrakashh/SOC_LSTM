`timescale 1ns/ 1ps
module top_4_by_1 #(
    parameter DATA_WIDTH = 12,
    parameter OUTPUT_WIDTH = 12,
    parameter FIFO_DEPTH = 16
)(
    input wire clk,
    input wire rst,
    input wire wr_en,
    input wire [DATA_WIDTH-1:0] data_r1, // row data rows of the matrix 
    input wire [DATA_WIDTH-1:0] weight_c1,weight_c2,weight_c3,weight_c4 ,// column of the matrix 
    output wire [OUTPUT_WIDTH-1:0] pe11,pe12,pe13,pe14,// processing element outputs
    output wire fifo_full,
    output wire fifo_empty,
    output wire fifo_valid
);
wire [DATA_WIDTH-1:0] ip_data_11,ip_data_21,ip_data_31,ip_data_41; // inputs to processing elements 
wire [DATA_WIDTH-1:0] weight_data_11,weight_data_12,weight_data_13,weight_data_14; //weights to processing elements 
wire [DATA_WIDTH-1:0] ip_pe11_to_pe12,ip_pe12_to_pe13,ip_pe13_to_pe14; //1st row forwarded data inputs
/* Instantiate FIFOs for data and weights */
wire pe11_ready = 1'b1;
wire pe12_ready = 1'b1;
wire pe13_ready = 1'b1;
wire pe14_ready = 1'b1;
// === Instantiate FIFOs with rd_en ===
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) data_fifo_11 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe11_ready),
    .data_in(data_r1), .data_out(ip_data_11),
    .full(fifo_full), .empty(fifo_empty), .valid(fifo_valid)
);

sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_11 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe11_ready),
    .data_in(weight_c1), .data_out(weight_data_11),
    .full(fifo_full), .empty(fifo_empty), .valid(fifo_valid)
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_12 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe12_ready),
    .data_in(weight_c2), .data_out(weight_data_12),
    .full(fifo_full), .empty(fifo_empty), .valid(fifo_valid)
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_13 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe13_ready),
    .data_in(weight_c3), .data_out(weight_data_13),
    .full(fifo_full), .empty(fifo_empty), .valid(fifo_valid)
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_14 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe14_ready),
    .data_in(weight_c4), .data_out(weight_data_14),
    .full(fifo_full), .empty(fifo_empty), .valid(fifo_valid)
);

/* Instantiate other processing elements (pe_12, pe_13, ..., pe_44) similarly, connecting forwarded data and weights appropriately */
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_11(
        .clk(clk),
        .rst(rst),
        .data_in(ip_data_11),
        .weight_in(weight_data_11),
        .output_reg(pe11),  
        .forwarded_data_out(ip_pe11_to_pe12),
        .forwarded_weight_out()
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) 
    pe_12(
        .clk(clk),
        .rst(rst),
        .data_in(ip_pe11_to_pe12),
        .weight_in(weight_data_12),
        .output_reg(pe12),  
        .forwarded_data_out(ip_pe12_to_pe13),
        .forwarded_weight_out()
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_13(
        .clk(clk),
        .rst(rst),
        .data_in(ip_pe12_to_pe13),
        .weight_in(weight_data_13),
        .output_reg(pe13),  
        .forwarded_data_out(ip_pe13_to_pe14),
        .forwarded_weight_out()
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_14(
        .clk(clk),
        .rst(rst),
        .data_in(ip_pe13_to_pe14),
        .weight_in(weight_data_14),
        .output_reg(pe14),  
        .forwarded_data_out(), // No further forwarding
        .forwarded_weight_out()
);    
                                                  
endmodule 
`timescale 1ns/ 1ps
module top_16_by_1 #(
    parameter DATA_WIDTH = 12,
    parameter OUTPUT_WIDTH = 12,
    parameter FIFO_DEPTH = 16
)(
    input wire clk,
    input wire rst,
    input wire wr_en,
    input wire [DATA_WIDTH-1:0] data_r1, // row data rows of the matrix 
    input wire [DATA_WIDTH-1:0] weight_c1,weight_c2,weight_c3,weight_c4 ,weight_c5,weight_c6,weight_c7,weight_c8,weight_c9,weight_c10,weight_c11,weight_c12,weight_c13,weight_c14,weight_c15,weight_c16,// column of the matrix 
    output wire [OUTPUT_WIDTH-1:0] pe1,pe2,pe3,pe4,pe5,pe6,pe7,pe8,pe9,pe10,pe11,pe12,pe13,pe14,pe15,pe16,// processing element outputs
    output wire fifo_full,
    output wire fifo_empty,
    output wire fifo_valid
);
wire [DATA_WIDTH-1:0] ip_data_11; //only one row of data input 
wire [DATA_WIDTH-1:0] weight_data_1,weight_data_2,weight_data_3,weight_data_4,weight_data_5,weight_data_6,weight_data_7,weight_data_8,weight_data_9,weight_data_10,weight_data_11,weight_data_12,weight_data_13,weight_data_14,weight_data_15,weight_data_16; //weights from fifo to processing elements 
wire [DATA_WIDTH-1:0] data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9,data_10,data_11,data_12,data_13,data_14,data_15,data_16; // data to processing elements only one row is there 

assign pe1_ready =1;
assign pe2_ready = 1;
assign pe2_ready = 1;
assign pe3_ready = 1;
assign pe4_ready = 1;
assign pe5_ready =1;
assign pe6_ready =1;
assign pe7_ready =1;
assign pe8_ready = 1;
assign pe9_ready =  1;
assign pe10_ready = 1;
assign pe11_ready = 1;
assign pe12_ready = 1;
assign pe13_ready = 1;
assign pe14_ready = 1;
assign pe15_ready = 1;
assign pe16_ready = 1;
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) data_fifo_1 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe1_ready),
    .data_in(data_r1), .data_out(ip_data_11),
    .full(fifo_full), .empty(fifo_empty), .valid(fifo_valid)
);

sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_1 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe1_ready),
    .data_in(weight_c1), .data_out(weight_data_1),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_2 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe2_ready),
    .data_in(weight_c2), .data_out(weight_data_2),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_3 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe3_ready),
    .data_in(weight_c3), .data_out(weight_data_3),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_4 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe4_ready),
    .data_in(weight_c4), .data_out(weight_data_4),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_5 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe5_ready),
    .data_in(weight_c5), .data_out(weight_data_5),
    .full(), .empty(), .valid()
);

sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_6 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe6_ready),
    .data_in(weight_c6), .data_out(weight_data_6),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_7 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe7_ready),
    .data_in(weight_c7), .data_out(weight_data_7),
    .full(), .empty(), .valid()
);  
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_8 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe8_ready),
    .data_in(weight_c8), .data_out(weight_data_8),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_9 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe9_ready),
    .data_in(weight_c9), .data_out(weight_data_9),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_10 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe10_ready),
    .data_in(weight_c10), .data_out(weight_data_10),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_11 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe11_ready),
    .data_in(weight_c11), .data_out(weight_data_11),
    .full(fifo_full), .empty(fifo_empty), .valid(fifo_valid)
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_12 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe12_ready),
    .data_in(weight_c12), .data_out(weight_data_12),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_13 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe13_ready),
    .data_in(weight_c13), .data_out(weight_data_13),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_14 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe14_ready),
    .data_in(weight_c14), .data_out(weight_data_14),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_15 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe15_ready),
    .data_in(weight_c15), .data_out(weight_data_15),
    .full(), .empty(), .valid()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_16 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe16_ready),
    .data_in(weight_c16), .data_out(weight_data_16),
    .full(), .empty(), .valid()
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_1 (
    .clk(clk),
    .rst(rst),
    .data_in(ip_data_11),
    .weight_in(weight_data_1),
    .output_reg(pe1),
    .forwarded_data_out(data_1)
);

processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_2 (
    .clk(clk),
    .rst(rst),
    .data_in(data_1),
    .weight_in(weight_data_2),
    .output_reg(pe2),
    .forwarded_data_out(data_2)
);

processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_3 (
    .clk(clk),
    .rst(rst),
    .data_in(data_2),
    .weight_in(weight_data_3),
    .output_reg(pe3),
    .forwarded_data_out(data_3)
);      
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_4 (
    .clk(clk),
    .rst(rst),
    .data_in(data_3),
    .weight_in(weight_data_4),
    .output_reg(pe4),
    .forwarded_data_out(data_4)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_5 (
    .clk(clk),
    .rst(rst),
    .data_in(data_4),
    .weight_in(weight_data_5),
    .output_reg(pe5),
    .forwarded_data_out(data_5)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_6 (
    .clk(clk),
    .rst(rst),
    .data_in(data_5),
    .weight_in(weight_data_6),
    .output_reg(pe6),
    .forwarded_data_out(data_6)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_7 (
    .clk(clk),
    .rst(rst),
    .data_in(data_6),
    .weight_in(weight_data_7),
    .output_reg(pe7),
    .forwarded_data_out(data_7)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_8 (
    .clk(clk),
    .rst(rst),
    .data_in(data_7),
    .weight_in(weight_data_8),
    .output_reg(pe8),
    .forwarded_data_out(data_8)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_9 (     
    .clk(clk),
    .rst(rst),
    .data_in(data_8),
    .weight_in(weight_data_9),
    .output_reg(pe9),
    .forwarded_data_out(data_9)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),    
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_10 (
    .clk(clk),  
    .rst(rst),
    .data_in(data_9),
    .weight_in(weight_data_10),
    .output_reg(pe10),
    .forwarded_data_out(data_10)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_11 (
    .clk(clk),
    .rst(rst),
    .data_in(data_10),
    .weight_in(weight_data_11),
    .output_reg(pe11),
    .forwarded_data_out(data_11)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_12 (        
    .clk(clk),
    .rst(rst),
    .data_in(data_11),
    .weight_in(weight_data_12),
    .output_reg(pe12),
    .forwarded_data_out(data_12)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_13 (
    .clk(clk),
    .rst(rst),
    .data_in(data_12),      
    .weight_in(weight_data_13),
    .output_reg(pe13),
    .forwarded_data_out(data_13)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_14 (
    .clk(clk),
    .rst(rst),
    .data_in(data_13),
    .weight_in(weight_data_14),
    .output_reg(pe14),
    .forwarded_data_out(data_14)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_15 (
    .clk(clk),
    .rst(rst),
    .data_in(data_14),
    .weight_in(weight_data_15),
    .output_reg(pe15),
    .forwarded_data_out(data_15)
);
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_16 (
    .clk(clk),
    .rst(rst),
    .data_in(data_15),
    .weight_in(weight_data_16),
    .output_reg(pe16),
    .forwarded_data_out(data_16)
);
endmodule
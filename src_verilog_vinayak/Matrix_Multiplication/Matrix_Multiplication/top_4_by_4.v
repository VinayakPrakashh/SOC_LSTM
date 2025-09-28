`timescale 1ns/ 1ps
module top_4_by_4 #(
    parameter DATA_WIDTH = 12,
    parameter OUTPUT_WIDTH = 12,
    parameter FIFO_DEPTH = 16
)(
    input wire clk,
    input wire rst,
    input wire wr_en,
    input wire [DATA_WIDTH-1:0] data_r1,data_r2,data_r3,data_r4, // row data rows of the matrix 
    input wire [DATA_WIDTH-1:0] weight_c1,weight_c2,weight_c3,weight_c4 ,// column of the matrix 
    output wire [OUTPUT_WIDTH-1:0] pe11,pe12,pe13,pe14,pe21,pe22,pe23,pe24,pe31,pe32,pe33,pe34,pe41,pe42,pe43,pe44, // processing element outputs
    output wire fifo_full,
    output wire fifo_empty,
    output wire fifo_valid
);
wire [DATA_WIDTH-1:0] ip_data_11,ip_data_21,ip_data_31,ip_data_41; // inputs to processing elements 
wire [DATA_WIDTH-1:0] weight_data_11,weight_data_12,weight_data_13,weight_data_14; //weights to processing elements 
wire [DATA_WIDTH-1:0] ip_pe11_to_pe12,ip_pe12_to_pe13,ip_pe13_to_pe14; //1st row forwarded data inputs
wire [DATA_WIDTH-1:0] ip_pe21_to_pe22,ip_pe22_to_pe23,ip_pe23_to_pe24; //2nd row forwarded data
wire [DATA_WIDTH-1:0] ip_pe31_to_pe32,ip_pe32_to_pe33,ip_pe33_to_pe34; //3rd row forwarded data
wire [DATA_WIDTH-1:0] ip_pe41_to_pe42,ip_pe42_to_pe43,ip_pe43_to_pe44; //4th row forwarded data  
wire [DATA_WIDTH-1:0]weight_pe11_to_pe21,weight_pe21_to_pe31,weight_pe31_to_pe41; //1st column forwarded weight inputs
wire [DATA_WIDTH-1:0]weight_pe12_to_pe22,weight_pe22_to_pe32,weight_pe32_to_pe42; //2nd column forwarded weight inputs
wire [DATA_WIDTH-1:0]weight_pe13_to_pe23,weight_pe23_to_pe33,weight_pe33_to_pe43; //3rd column forwarded weight inputs
wire [DATA_WIDTH-1:0]weight_pe14_to_pe24,weight_pe24_to_pe34,weight_pe34_to_pe44; //4th column forwarded weight inputs

/* Instantiate FIFOs for data and weights */
wire pe11_ready = 1'b1;
wire pe21_ready = 1'b1;
wire pe31_ready = 1'b1;
wire pe41_ready = 1'b1;

wire pe12_ready = 1'b1;
wire pe22_ready = 1'b1;
wire pe32_ready = 1'b1;
wire pe42_ready = 1'b1;

wire pe13_ready = 1'b1;
wire pe23_ready = 1'b1;
wire pe33_ready = 1'b1;
wire pe43_ready = 1'b1;

wire pe14_ready = 1'b1;
wire pe24_ready = 1'b1;
wire pe34_ready = 1'b1;
wire pe44_ready = 1'b1;
// === Instantiate FIFOs with rd_en ===
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) data_fifo_11 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe11_ready),
    .data_in(data_r1), .data_out(ip_data_11),
    .full(fifo_full), .empty(fifo_empty), .valid(fifo_valid)
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) data_fifo_21 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe21_ready),
    .data_in(data_r2), .data_out(ip_data_21),
    .full(fifo_full), .empty(fifo_empty), .valid(fifo_valid)
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) data_fifo_31 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe31_ready),
    .data_in(data_r3), .data_out(ip_data_31),
    .full(fifo_full), .empty(fifo_empty), .valid(fifo_valid)
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) data_fifo_41 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe41_ready),
    .data_in(data_r4), .data_out(ip_data_41),
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
        .forwarded_weight_out(weight_pe11_to_pe21)
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
        .forwarded_weight_out(weight_pe12_to_pe22)
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
        .forwarded_weight_out(weight_pe13_to_pe23)
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
        .forwarded_weight_out(weight_pe14_to_pe24)
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_21(
        .clk(clk),
        .rst(rst),
        .data_in(ip_data_21),
        .weight_in(weight_pe11_to_pe21),
        .output_reg(pe21),  
        .forwarded_data_out(ip_pe21_to_pe22),
        .forwarded_weight_out(weight_pe21_to_pe31)
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),        
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_22(
        .clk(clk),
        .rst(rst),
        .data_in(ip_pe21_to_pe22),
        .weight_in(weight_pe12_to_pe22),
        .output_reg(pe22),  
        .forwarded_data_out(ip_pe22_to_pe23),
        .forwarded_weight_out(weight_pe22_to_pe32)
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_23(
        .clk(clk),
        .rst(rst),
        .data_in(ip_pe22_to_pe23),
        .weight_in(weight_pe13_to_pe23),            
        .output_reg(pe23),  
        .forwarded_data_out(ip_pe23_to_pe24),
        .forwarded_weight_out(weight_pe23_to_pe33) 
);
processing_element #(    
    .DATA_WIDTH(DATA_WIDTH),                    
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_24(
        .clk(clk),
        .rst(rst),
        .data_in(ip_pe23_to_pe24),
        .weight_in(weight_pe14_to_pe24),            
        .output_reg(pe24),  
        .forwarded_data_out(), // No further forwarding
        .forwarded_weight_out(weight_pe24_to_pe34)  
    ) ;
    processing_element #(       
    .DATA_WIDTH(DATA_WIDTH), 
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_31(        
        .clk(clk),
        .rst(rst),
        .data_in(ip_data_31),
        .weight_in(weight_pe21_to_pe31),
        .output_reg(pe31),  
        .forwarded_data_out(ip_pe31_to_pe32),
        .forwarded_weight_out(weight_pe31_to_pe41)
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_32(                                
        .clk(clk),
        .rst(rst),
        .data_in(ip_pe31_to_pe32),
        .weight_in(weight_pe22_to_pe32),
        .output_reg(pe32),  
        .forwarded_data_out(ip_pe32_to_pe33),
        .forwarded_weight_out(weight_pe32_to_pe42)
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_33(                            
        .clk(clk),  
        .rst(rst),
        .data_in(ip_pe32_to_pe33),
        .weight_in(weight_pe23_to_pe33),        
        .output_reg(pe33), 
        .forwarded_data_out(ip_pe33_to_pe34),
        .forwarded_weight_out(weight_pe33_to_pe43)
);    
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_34(                            
        .clk(clk),  
        .rst(rst),
        .data_in(ip_pe33_to_pe34),
        .weight_in(weight_pe24_to_pe34),        
        .output_reg(pe34), 
        .forwarded_data_out(), // No further forwarding
        .forwarded_weight_out(weight_pe34_to_pe44)
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_41(                
        .clk(clk),                                      
        .rst(rst),
        .data_in(ip_data_41),   
        .weight_in(weight_pe31_to_pe41),
        .output_reg(pe41),  
        .forwarded_data_out(ip_pe41_to_pe42),
        .forwarded_weight_out() // No further forwarding
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_42(                                
        .clk(clk),
        .rst(rst),
        .data_in(ip_pe41_to_pe42),
        .weight_in(weight_pe32_to_pe42),
        .output_reg(pe42),          
        .forwarded_data_out(ip_pe42_to_pe43),       
        .forwarded_weight_out() // No further forwarding
);
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),    
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_43(                            
        .clk(clk),  
        .rst(rst),
        .data_in(ip_pe42_to_pe43),
        .weight_in(weight_pe33_to_pe43),        
        .output_reg(pe43), 
        .forwarded_data_out(ip_pe43_to_pe44),
        .forwarded_weight_out() );// No further forwarding  
processing_element #(       
    .DATA_WIDTH(DATA_WIDTH),  
    .OUTPUT_WIDTH(OUTPUT_WIDTH) ) pe_44(                            
        .clk(clk),  
        .rst(rst),
        .data_in(ip_pe43_to_pe44),
        .weight_in(weight_pe34_to_pe44),        
        .output_reg(pe44), 
        .forwarded_data_out(), // No further forwarding
        .forwarded_weight_out() // No further forwarding
);                                                          
endmodule 
`timescale 1ns/ 1ps
module top_16_by_1 #(
    parameter DATA_WIDTH = 12,
    parameter OUTPUT_WIDTH = 12,
    parameter FIFO_DEPTH = 16
)(
    input wire clk,
    input wire rst,
    input wire wr_en,
    input wire wr_en_data,
    input wire [DATA_WIDTH-1:0] data_r1, // row data rows of the matrix 
    input wire [DATA_WIDTH-1:0] weight_c1,weight_c2,weight_c3,weight_c4 ,weight_c5,weight_c6,weight_c7,weight_c8,weight_c9,weight_c10,weight_c11,weight_c12,weight_c13,weight_c14,weight_c15,weight_c16,// column of the matrix 
    output wire [OUTPUT_WIDTH-1:0] pe1,pe2,pe3,pe4,pe5,pe6,pe7,pe8,pe9,pe10,pe11,pe12,pe13,pe14,pe15,pe16,// processing element outputs
    output wire fifo_full,
    output wire fifo_empty
);

wire [DATA_WIDTH-1:0]pe1_reg,pe2_reg,pe3_reg,pe4_reg,pe5_reg,pe6_reg,pe7_reg,pe8_reg,pe9_reg,pe10_reg,pe11_reg,pe12_reg,pe13_reg,pe14_reg,pe15_reg,pe16_reg;
wire [DATA_WIDTH-1:0] ip_data_11; //only one row of data input 
wire [DATA_WIDTH-1:0] weight_data_1,weight_data_2,weight_data_3,weight_data_4,weight_data_5,weight_data_6,weight_data_7,weight_data_8,weight_data_9,weight_data_10,weight_data_11,weight_data_12,weight_data_13,weight_data_14,weight_data_15,weight_data_16; //weights from fifo to processing elements 
wire [DATA_WIDTH-1:0] data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9,data_10,data_11,data_12,data_13,data_14,data_15,data_16; // data to processing elements only one row is there 

// FIXED: Simple control logic - only track reading state
// ...existing code...

// FIXED: Enhanced control logic to handle 1-cycle pipeline delay
reg reading;
reg output_valid;
reg [4:0] read_count;  // Count how many values have been read (0-15)

always @(posedge clk or posedge rst) begin
    if (rst) begin
        reading <= 0;
        output_valid <= 0;
        read_count <= 0;
    end
    else if (fifo_full && !reading) begin
        reading <= 1;           // Start reading when FIFO becomes full
        output_valid <= 0;      // Output not valid yet (1 cycle delay)
        read_count <= 0;        // Reset counter
    end
    else if (reading) begin
        output_valid <= 1;      // Output becomes valid after 1 cycle
        if (read_count < 16) begin
            read_count <= read_count + 1;  // Count each read
        end else begin
            reading <= 0;       // Stop reading after 16 values (0-15)
            read_count <= 0;    // Reset counter
        end
    end
    else if (output_valid && !reading) begin
        // Continue showing outputs for 1 more cycle after reading stops
        output_valid <= 0;
    end
end

// FIXED: Synchronized read enable for all FIFOs
wire rd_en = reading;

// All pe_ready signals use the same synchronized control
wire pe1_ready,pe2_ready,pe3_ready,pe4_ready,pe5_ready,pe6_ready,pe7_ready,pe8_ready,pe9_ready,pe10_ready,pe11_ready,pe12_ready,pe13_ready,pe14_ready,pe15_ready,pe16_ready;

assign pe1_ready  = rd_en;
assign pe2_ready  = rd_en;
assign pe3_ready  = rd_en;
assign pe4_ready  = rd_en;
assign pe5_ready  = rd_en;
assign pe6_ready  = rd_en;
assign pe7_ready  = rd_en;
assign pe8_ready  = rd_en;
assign pe9_ready  = rd_en;
assign pe10_ready = rd_en;
assign pe11_ready = rd_en;
assign pe12_ready = rd_en;
assign pe13_ready = rd_en;
assign pe14_ready = rd_en;
assign pe15_ready = rd_en;
assign pe16_ready = rd_en;

// FIXED: Output control considering pipeline delay
assign pe1  = (reading || output_valid) ? pe1_reg  : {OUTPUT_WIDTH{1'b0}};
assign pe2  = (reading || output_valid) ? pe2_reg  : {OUTPUT_WIDTH{1'b0}};
assign pe3  = (reading || output_valid) ? pe3_reg  : {OUTPUT_WIDTH{1'b0}};
assign pe4  = (reading || output_valid) ? pe4_reg  : {OUTPUT_WIDTH{1'b0}};
assign pe5  = (reading || output_valid) ? pe5_reg  : {OUTPUT_WIDTH{1'b0}};
assign pe6  = (reading || output_valid) ? pe6_reg  : {OUTPUT_WIDTH{1'b0}};
assign pe7  = (reading || output_valid) ? pe7_reg  : {OUTPUT_WIDTH{1'b0}};
assign pe8  = (reading || output_valid) ? pe8_reg  : {OUTPUT_WIDTH{1'b0}};
assign pe9  = (reading || output_valid) ? pe9_reg  : {OUTPUT_WIDTH{1'b0}};
assign pe10 = (reading || output_valid) ? pe10_reg : {OUTPUT_WIDTH{1'b0}};
assign pe11 = (reading || output_valid) ? pe11_reg : {OUTPUT_WIDTH{1'b0}};
assign pe12 = (reading || output_valid) ? pe12_reg : {OUTPUT_WIDTH{1'b0}};
assign pe13 = (reading || output_valid) ? pe13_reg : {OUTPUT_WIDTH{1'b0}};
assign pe14 = (reading || output_valid) ? pe14_reg : {OUTPUT_WIDTH{1'b0}};
assign pe15 = (reading || output_valid) ? pe15_reg : {OUTPUT_WIDTH{1'b0}};
assign pe16 = (reading || output_valid) ? pe16_reg : {OUTPUT_WIDTH{1'b0}};

// ...rest of the module remains the same...

// CRITICAL FIX: Only weight_fifo_16 drives fifo_full and fifo_empty to avoid signal conflicts
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) data_fifo_1 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en_data), .rd_en(pe1_ready),
    .data_in(data_r1), .data_out(ip_data_11),
    .full(),.empty()  // DISCONNECTED to avoid signal conflict
);

sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_1 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe1_ready),
    .data_in(weight_c1), .data_out(weight_data_1),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_2 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe2_ready),
    .data_in(weight_c2), .data_out(weight_data_2),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_3 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe3_ready),
    .data_in(weight_c3), .data_out(weight_data_3),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_4 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe4_ready),
    .data_in(weight_c4), .data_out(weight_data_4),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_5 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe5_ready),
    .data_in(weight_c5), .data_out(weight_data_5),
    .full(),.empty()
);

sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_6 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe6_ready),
    .data_in(weight_c6), .data_out(weight_data_6),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_7 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe7_ready),
    .data_in(weight_c7), .data_out(weight_data_7),
    .full(),.empty()
);  
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_8 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe8_ready),
    .data_in(weight_c8), .data_out(weight_data_8),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_9 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe9_ready),
    .data_in(weight_c9), .data_out(weight_data_9),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_10 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe10_ready),
    .data_in(weight_c10), .data_out(weight_data_10),
    .full(),.empty()
);
    
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_11 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe11_ready),
    .data_in(weight_c11), .data_out(weight_data_11),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_12 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe12_ready),
    .data_in(weight_c12), .data_out(weight_data_12),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_13 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe13_ready),
    .data_in(weight_c13), .data_out(weight_data_13),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_14 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe14_ready),
    .data_in(weight_c14), .data_out(weight_data_14),
    .full(),.empty()
);
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_15 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe15_ready),
    .data_in(weight_c15), .data_out(weight_data_15),
    .full(),.empty()
);

// ONLY weight_fifo_16 drives the status signals (as per your design intent)
sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(FIFO_DEPTH)) weight_fifo_16 (
    .clk(clk), .rst(rst),
    .wr_en(wr_en), .rd_en(pe16_ready),
    .data_in(weight_c16), .data_out(weight_data_16),
    .full(fifo_full),.empty(fifo_empty)
);

// All processing elements remain the same...
// (Include all 16 processing elements as before)

// All processing elements remain the same
processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_WIDTH(OUTPUT_WIDTH)
) pe_1 (
    .clk(clk),
    .rst(rst),
    .data_in(ip_data_11),
    .weight_in(weight_data_1),
    .output_reg(pe1_reg),
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
    .output_reg(pe2_reg),
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
    .output_reg(pe3_reg),
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
    .output_reg(pe4_reg),
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
    .output_reg(pe5_reg),
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
    .output_reg(pe6_reg),
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
    .output_reg(pe7_reg),
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
    .output_reg(pe8_reg),
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
    .output_reg(pe9_reg),
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
    .output_reg(pe10_reg),
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
    .output_reg(pe11_reg),
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
    .output_reg(pe12_reg),
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
    .output_reg(pe13_reg),
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
    .output_reg(pe14_reg),
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
    .output_reg(pe15_reg),
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
    .output_reg(pe16_reg),
    .forwarded_data_out(data_16)
);
//------------------------------------------
endmodule
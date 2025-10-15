module main #(
    parameter TILE_SIZE = 256,
    parameter DATA_WIDTH = 12,
    parameter NUM_LUTRAMS = 32,
    parameter ADDR_WIDTH = 4
) (
    input wire clk,
    input wire rst,
    input wire wr_en,
    input wire [DATA_WIDTH-1:0] data_in,data_in_2,
    input wire [ADDR_WIDTH-1:0] wr_addr,
    input wire [ADDR_WIDTH-1:0] rd_addr); // 5 bits to select one of 32 LUTRAMs
    // FIXED: Individual outputs instead of arra
    // FIXED: Individual wires for demux outputs
    reg [3:0]lut_write_counter;
    wire [DATA_WIDTH-1:0] demux_out_0, demux_out_1, demux_out_2, demux_out_3;
    wire [DATA_WIDTH-1:0] demux_out_4, demux_out_5, demux_out_6, demux_out_7;
    wire [DATA_WIDTH-1:0] demux_out_8, demux_out_9, demux_out_10, demux_out_11;
    wire [DATA_WIDTH-1:0] demux_out_12, demux_out_13, demux_out_14, demux_out_15;

    // Output from each LUTRAM
    wire [DATA_WIDTH-1:0] data_out_0, data_out_1, data_out_2, data_out_3,data_out_4 ,data_out_5,data_out_6,data_out_7;
    wire [DATA_WIDTH-1:0] data_out_8, data_out_9, data_out_10, data_out_11,data_out_12 ,data_out_13,data_out_14,data_out_15;
    // Individual write enables
    wire [31:0] lutram_wr_en;
    reg [5:0]lutram_select;
    // Generate write enable signals
    assign lutram_wr_en[0]  = wr_en && (lutram_select == 5'd0);
    assign lutram_wr_en[1]  = wr_en && (lutram_select == 5'd1);
    assign lutram_wr_en[2]  = wr_en && (lutram_select == 5'd2);
    assign lutram_wr_en[3]  = wr_en && (lutram_select == 5'd3);
    assign lutram_wr_en[4]  = wr_en && (lutram_select == 5'd4);
    assign lutram_wr_en[5]  = wr_en && (lutram_select == 5'd5);
    assign lutram_wr_en[6]  = wr_en && (lutram_select == 5'd6);
    assign lutram_wr_en[7]  = wr_en && (lutram_select == 5'd7);
    assign lutram_wr_en[8]  = wr_en && (lutram_select == 5'd8);
    assign lutram_wr_en[9]  = wr_en && (lutram_select == 5'd9);
    assign lutram_wr_en[10] = wr_en && (lutram_select == 5'd10);
    assign lutram_wr_en[11] = wr_en && (lutram_select == 5'd11);
    assign lutram_wr_en[12] = wr_en && (lutram_select == 5'd12);
    assign lutram_wr_en[13] = wr_en && (lutram_select == 5'd13);
    assign lutram_wr_en[14] = wr_en && (lutram_select == 5'd14);
    assign lutram_wr_en[15] = wr_en && (lutram_select == 5'd15);
    //Outputs of each processing element in matrix multiplication module 
    wire [DATA_WIDTH-1:0] result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8,result_9,result_10,result_11,result_12,result_13,result_14,result_15,result_16;
    // FIXED: Properly connect to your demux module
    demux_1to16 #(
        .DATA_WIDTH(DATA_WIDTH)
    ) data_demux (
        .data_in(data_in),
        .select(lutram_select),
        .data_out_0(demux_out_0),
        .data_out_1(demux_out_1),
        .data_out_2(demux_out_2),
        .data_out_3(demux_out_3),
        .data_out_4(demux_out_4),
        .data_out_5(demux_out_5),
        .data_out_6(demux_out_6),
        .data_out_7(demux_out_7),
        .data_out_8(demux_out_8),
        .data_out_9(demux_out_9),
        .data_out_10(demux_out_10),
        .data_out_11(demux_out_11),
        .data_out_12(demux_out_12),
        .data_out_13(demux_out_13),
        .data_out_14(demux_out_14),
        .data_out_15(demux_out_15),
       
    );

    // 32 LUTRAM instances
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_0 
        (.clk(clk), .wr_en(lutram_wr_en[0]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_0), .data_out(data_out_0));
    
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_1 
        (.clk(clk), .wr_en(lutram_wr_en[1]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_1), .data_out(data_out_1));
    
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_2 
        (.clk(clk), .wr_en(lutram_wr_en[2]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_2), .data_out(data_out_2));
    

    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_3 
        (.clk(clk), .wr_en(lutram_wr_en[3]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_3), .data_out(data_out_3));

    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_4 
        (.clk(clk), .wr_en(lutram_wr_en[4]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_4), .data_out(data_out_4));   
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_5 
        (.clk(clk), .wr_en(lutram_wr_en[5]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_5), .data_out(data_out_5));
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_6 
        (.clk(clk), .wr_en(lutram_wr_en[6]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_6), .data_out(data_out_6));
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_7 
        (.clk(clk), .wr_en(lutram_wr_en[7]), .wr_addr(wr_addr), .           rd_addr(rd_addr), .data_in(demux_out_7), .data_out(data_out_7));
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_8 
        (.clk(clk), .wr_en(lutram_wr_en[8]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_8), .data_out(data_out_8));
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_9 
        (.clk(clk), .wr_en(lutram_wr_en[9]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_9), .data_out(data_out_9));
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_10 
        (.clk(clk), .wr_en(lutram_wr_en[10]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_10), .data_out(data_out_10));
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_11 
        (.clk(clk), .wr_en(lutram_wr_en[11]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_11), .data_out(data_out_11));
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_12 
        (.clk(clk), .wr_en(lutram_wr_en[12]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_12), .data_out(data_out_12));
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_13 
        (.clk(clk), .wr_en(lutram_wr_en[13]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_13), .data_out(data_out_13));
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH))       lutram_14 
        (.clk(clk), .wr_en(lutram_wr_en[14]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_14), .data_out(data_out_14));
    lutram_16 #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) lutram_15 
        (.clk(clk), .wr_en(lutram_wr_en[15]), .wr_addr(wr_addr), .rd_addr(rd_addr), .data_in(demux_out_15), .data_out(data_out_15));

   // systolic array multiplier module 
    top_16_by_1 matmul (
        .clk(clk),
        .rst(rst),
        .weight_c1(data_out_0), 
        .weight_c2(data_out_1),
        .weight_c3(data_out_2),
        .weight_c4(data_out_3),
        .weight_c5(data_out_4),
        .weight_c6(data_out_5),
        .weight_c7(data_out_6),
        .weight_c8(data_out_7),
        .weight_c9(data_out_8),
        .weight_c10(data_out_9),
        .weight_c11(data_out_10),
        .weight_c12(data_out_11),
        .weight_c13(data_out_12),
        .weight_c14(data_out_13),   
        .weight_c15(data_out_14),
        .weight_c16(data_out_15),
        .data_in(data_in_2).
        .pe1(result_1),
        .pe2(result_2),
        .pe3(result_3),
        .pe4(result_4),
        .pe5(result_5),
        .pe6(result_6),
        .pe7(result_7),
        .pe8(result_8),
        .pe9(result_9),
        .pe10(result_10),
        .pe11(result_11),
        .pe12(result_12),
        .pe13(result_13),
        .pe14(result_14),
        .pe15(result_15),
        .pe16(result_16)
    );

always @(posedge clk or posedge rst) begin
    if (rst) begin
        lut_write_counter <= 4'd0;
        lutram_select <= 5'd0;
    end else if (wr_en) begin
        lut_write_counter <= lut_write_counter + 4'd1;
        if (lut_write_counter == 4'd15) begin
            lut_write_counter <= 4'd0; // Reset counter after reaching 15
            lut_ram_select <= lutram_select + 5'd1; // Move to next LUTRAM
            if (lut_ram_select == 5'd15) begin
                lut_ram_select <= 5'd0; // Wrap around to first LUTRAM
            end
        end
    end
end

endmodule
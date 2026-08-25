module sync_fifo #(
    parameter DATA_WIDTH = 80,
    parameter DEPTH = 20,
    parameter ADDR_WIDTH = 5,
    parameter INIT_COUNT = 20
)(
    input wire clk,
    input wire rst_n,
    input wire wr_en,
    input wire rd_en,
    input wire [DATA_WIDTH-1:0] wr_data,
    output reg [DATA_WIDTH-1:0] rd_data,
    output wire full,
    output wire empty
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_WIDTH-1:0] wr_ptr;
    reg [ADDR_WIDTH-1:0] rd_ptr;
    reg [ADDR_WIDTH-1:0] count;
    
    assign full = (count == DEPTH);
    assign empty = (count == 0);
    initial begin
    mem[0]  = 80'h80A2_0063_01A9_0066_801E;
    mem[1]  = 80'h80A2_0063_01A9_0066_801E;
    mem[2]  = 80'h80A2_0063_01A9_0066_801E;
    mem[3]  = 80'h80A2_0063_01A9_0065_801E;
    mem[4]  = 80'h80A2_0063_01A9_0066_801E;
    mem[5]  = 80'h80A2_0063_01A9_0066_801E;
    mem[6]  = 80'h80A2_0063_01A9_0065_801E;
    mem[7]  = 80'h80A2_0063_01A9_0065_801E;
    mem[8]  = 80'h80A2_0051_01A9_0054_8020;
    mem[9]  = 80'h80A2_001E_01A9_0023_8027;
    mem[10] = 80'h80A3_8046_01A9_803F_8034;
    mem[11] = 80'h80A3_8055_01A9_804D_8037;
    mem[12] = 80'h80A3_807D_01A9_8074_803E;
    mem[13] = 80'h80A3_804F_01A9_8047_8039;
    mem[14] = 80'h80A3_804B_01A9_8044_8039;
    mem[15] = 80'h80A4_80D7_01A9_80CD_804C;
    mem[16] = 80'h80A4_80F5_01A6_80EB_8052;
    mem[17] = 80'h80A4_80A4_01A6_809C_804A;
    mem[18] = 80'h80A4_8094_01A6_808C_8048;
    mem[19] = 80'h80A5_8090_01A6_8088_8049;
end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= INIT_COUNT;
            
            // Format: [Voltage][Current][Temp][Power][Capacity]
            // Sign-magnitude Q7.8: bit[15]=sign, bits[14:0]=magnitude
            
//mem[0]  <= 80'h81E3_028C_8036_00B8_038D;
//mem[1]  <= 80'h81E4_0AED_8036_02FF_03A5;
//mem[2]  <= 80'h81E4_0AF3_8036_02FF_03A8;
//mem[3]  <= 80'h81E3_0AFA_8036_02FF_03A9;
//mem[4]  <= 80'h81E3_07CC_8036_022C_03A0;
//mem[5]  <= 80'h81E3_0192_8036_0071_038E;
//mem[6]  <= 80'h81E3_802E_8036_800D_038A;
//mem[7]  <= 80'h81E3_8061_8036_801B_038A;
//mem[8]  <= 80'h81E3_8068_8036_801C_0389;
//mem[9]  <= 80'h81E3_806F_8036_801D_0389;
//mem[10] <= 80'h81E3_8068_8036_801C_0389;
//mem[11] <= 80'h81E3_806F_8036_801D_0389;
//mem[12] <= 80'h81E3_8068_8036_801C_0389;
//mem[13] <= 80'h81E3_806F_8036_801D_0389;
//mem[14] <= 80'h81E3_806F_8036_801D_0389;
//mem[15] <= 80'h81E3_808B_8036_801F_0389;
//mem[16] <= 80'h81E3_80C9_804F_802C_0388;
//mem[17] <= 80'h81E3_80EA_8036_8035_0388;
//mem[18] <= 80'h81E4_837B_8036_8087_0385;
//mem[19] <= 80'h81E6_8F4B_8036_80F5_0377;
            
        end else if (wr_en && !full) begin
            mem[wr_ptr] <= wr_data;
            wr_ptr <= (wr_ptr == DEPTH-1) ? 0 : wr_ptr + 1;
        end
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= 0;
            rd_data <= 0;
        end else if (rd_en && !empty) begin
            rd_data <= mem[rd_ptr];
            rd_ptr <= (rd_ptr == DEPTH-1) ? 0 : rd_ptr + 1;
        end
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= INIT_COUNT;
        end else begin
            case ({wr_en && !full, rd_en && !empty})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: count <= count;
            endcase
        end
    end

endmodule

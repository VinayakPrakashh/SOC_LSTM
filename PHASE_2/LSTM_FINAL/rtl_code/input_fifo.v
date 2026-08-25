`timescale 1ns/1ps

module sync_fifo #(
    parameter DATA_WIDTH = 80,
    parameter DEPTH      = 20,
    parameter ADDR_WIDTH = 5,
    parameter INIT_COUNT = 20
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  wr_en,
    input  wire                  rd_en,
    input  wire [DATA_WIDTH-1:0] wr_data,
    output reg  [DATA_WIDTH-1:0] rd_data,
    output wire                  full,
    output wire                  empty
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_WIDTH-1:0] wr_ptr;
    reg [ADDR_WIDTH-1:0] rd_ptr;
    reg [ADDR_WIDTH-1:0] count;

    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    // Pre-loaded data - intentional
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
    // ? Write - mem separate, no async reset on array
    always @(posedge clk) begin
        if (wr_en && !full)
            mem[wr_ptr] <= wr_data;
    end

    // ? Write pointer - reset to 0 (all slots pre-filled, wrap correctly)
    always @(posedge clk) begin
        if (!rst_n)
            wr_ptr <= {ADDR_WIDTH{1'b0}};  // FIX: was INIT_COUNT=20 (out of bounds!)
        else if (wr_en && !full)
            wr_ptr <= (wr_ptr == DEPTH-1) ? 0 : wr_ptr + 1;
    end

    // ? Read - no reset on rd_data (prevents RAM dissolution)
    always @(posedge clk) begin
        if (!rst_n)
            rd_ptr <= {ADDR_WIDTH{1'b0}};
        else if (rd_en && !empty) begin
            rd_data <= mem[rd_ptr];
            rd_ptr  <= (rd_ptr == DEPTH-1) ? 0 : rd_ptr + 1;
        end
    end

    // ? Count - reset to INIT_COUNT (correct, FIFO starts full)
    always @(posedge clk) begin
        if (!rst_n)
            count <= INIT_COUNT;  // intentional - starts full
        else begin
            case ({wr_en && !full, rd_en && !empty})
                2'b10:   count <= count + 1;
                2'b01:   count <= count - 1;
                default: count <= count;
            endcase
        end
    end

endmodule
module dual_bram #(
    parameter DATA_WIDTH = 12,
    parameter TILE_SIZE  = 256
)(
    input  wire                       clk,
    input  wire                       rst,

    // Port A (Read/Write)
    input  wire                       wr_en_a,
    input  wire [$clog2(TILE_SIZE)-1:0] addr_a,
    input  wire [DATA_WIDTH-1:0]      data_in_a,
    output reg  [DATA_WIDTH-1:0]      data_out_a,

    // Port B (Read Only)
    input  wire [$clog2(TILE_SIZE)-1:0] addr_b,
    output reg  [DATA_WIDTH-1:0]      data_out_b
);

    // True dual-port block RAM
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem [0:TILE_SIZE-1];

    // Port A: Read/Write
    always @(posedge clk) begin
        if (wr_en_a)
            mem[addr_a] <= data_in_a;   // Write
        data_out_a <= mem[addr_a];      // Read
    end

    // Port B: Independent Read
    always @(posedge clk) begin
        data_out_b <= mem[addr_b];      // Read-only port
    end

endmodule

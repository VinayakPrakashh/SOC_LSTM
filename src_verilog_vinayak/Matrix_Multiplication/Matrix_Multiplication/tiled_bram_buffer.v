module tiled_bram_buffer#(
    parameter DATA_WIDTH = 12,
    parameter TILE_SIZE  = 16
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   wr_en,
    input  wire [$clog2(TILE_SIZE)-1:0] addr, // address for read/write
    input  wire [DATA_WIDTH-1:0]  data_in,
    output reg  [DATA_WIDTH-1:0]  data_out
);

    // Distributed LUTRAM memory
    (* ram_style = "distributed" *) reg [DATA_WIDTH-1:0] mem [0:TILE_SIZE-1];

    always @(posedge clk) begin
        if (wr_en) begin
            mem[addr] <= data_in;// write
        end
       
    end
 assign  data_out = mem[addr];     // read
endmodule
  
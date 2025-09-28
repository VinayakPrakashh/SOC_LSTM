module tiling_top #(
    parameter TILE_SIZE=4,
    parameter MAX_SIZE=512,
    parameter DATA_WIDTH=32,
)(  input clk,rst,wr_en,
    input [DATA_WIDTH-1:0] data_in,
    output reg [7:0] counter
);
wire [DATA_WIDTH-1:0] data_to_tile;
global_bram main_bram(.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .wr_en(wr_en),
                      .counter(counter)
                      );
tiled_bram ip_bram_1 (.clk(clk),.
                      .data_in(data_in),
                     .wr_en(),
                     .data_out(data_out)
                     );
tiled_bram ip_bram_2 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out)
                      .wr_en());
tiled_bram ip_bram_3 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_4 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_5 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_6 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_7 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_8 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_9 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_10 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_11 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_12 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_13 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_14 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_15 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram ip_bram_16 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_1 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_2 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_3 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_4 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_5 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_6 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_7 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_8 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_9 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_10 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_11 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_12 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_13 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());    
tiled_bram weight_bram_14 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());                      
tiled_bram weight_bram_15 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());
tiled_bram weight_bram_16 (.clk(clk),
                      .rst(rst),
                      .data_in(data_in),
                      .data_out(data_out),
                      .wr_en());                      
                      


endmodule
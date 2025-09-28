module tiled_bram #(
    parameter DATA_WIDTH = 12,
    parameter TILE_SIZE  = 4
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   wr_en,
    input  wire [DATA_WIDTH-1:0]  data_in,
    output reg  [DATA_WIDTH-1:0]  data_out
);

    reg [7:0] counter;
    reg [DATA_WIDTH-1:0] local_tile_mem [0:TILE_SIZE-1];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            counter  <= 0;
            data_out <= 0;
        end else begin
            if (wr_en) begin
                local_tile_mem[counter] <= data_in;
                counter <= counter + 1;
            end
            data_out <= local_tile_mem[counter];
        end
    end

endmodule

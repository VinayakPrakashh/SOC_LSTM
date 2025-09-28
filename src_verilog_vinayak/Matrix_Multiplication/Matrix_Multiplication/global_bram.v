module global_bram #(
    parameter DATA_WIDTH = 12,
    parameter MAX_INPUT_SIZE=512
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   wr_en,
    input  wire [DATA_WIDTH-1:0]  data_in,
    output reg  [DATA_WIDTH-1:0]  data_out,
    output reg [7:0] counter,
);
    reg [DATA_WIDTH-1:0] global_mem [0:MAX_INPUT_SIZE];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            counter  <= 0;
            data_out <= 0;
        end else begin
            if (wr_en) begin
                 global_mem[counter]<= data_in;
                counter <= counter + 1;
            end
            data_out <= local_tile_mem[counter];
        end
    end

endmodule

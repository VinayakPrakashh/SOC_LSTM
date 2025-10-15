module lutram_16 #(
    parameter DATA_WIDTH = 12,
    parameter ADDR_WIDTH = 4  // 2^4 = 16 elements
)(
    input wire clk,
    input wire wr_en,
    input wire [ADDR_WIDTH-1:0] wr_addr,
    input wire [ADDR_WIDTH-1:0] rd_addr,
    input wire [DATA_WIDTH-1:0] data_in,
    output reg [DATA_WIDTH-1:0] data_out
);

    // 16-element memory array
    reg [DATA_WIDTH-1:0] ram_array [0:15];
    
    // Write operation (synchronous)
    always @(posedge clk) begin
        if (wr_en) begin
            ram_array[wr_addr] <= data_in;
        end
    end
    
    // Read operation (asynchronous for LUTRAM behavior)
    always @(*) begin
        data_out = ram_array[rd_addr];
    end

endmodule
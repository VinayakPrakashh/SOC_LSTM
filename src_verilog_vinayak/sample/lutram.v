module lutram_simple #(
    parameter DATA_WIDTH = 12,
    parameter ADDR_BITS = 4,     // 64 locations (typical for LUTRAM)
    parameter DEPTH = 16
) (
    input clk,
    input we,
    input [ADDR_BITS-1:0] addr,
    input [DATA_WIDTH-1:0] data_in,
    output reg [DATA_WIDTH-1:0] data_out
);

    // LUTRAM array - will be synthesized to distributed RAM
    (* ram_style = "distributed" *) reg [DATA_WIDTH-1:0] lutram [0:DEPTH-1];
    
    // Write operation
    always @(posedge clk) begin
        if (we) begin
            lutram[addr] <= data_in;
        end
    end
    
    // Read operation (combinational for LUTRAM)
    always @(*) begin
        data_out = lutram[addr];
    end

endmodule
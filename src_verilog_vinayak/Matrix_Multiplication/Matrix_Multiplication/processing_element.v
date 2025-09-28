module processing_element #(
    parameter DATA_WIDTH = 12,
    parameter OUTPUT_WIDTH = 12
)(
    input  wire clk,
    input  wire rst,
    input  wire [DATA_WIDTH-1:0] data_in,
    input  wire [DATA_WIDTH-1:0] weight_in,
    output reg  [OUTPUT_WIDTH-1:0] output_reg,
    output wire [DATA_WIDTH-1:0]forwarded_data_out,forwarded_weight_out
);

    reg [DATA_WIDTH-1:0] data_reg, weight_reg;
    reg [OUTPUT_WIDTH-1:0] product;
    reg [OUTPUT_WIDTH-1:0] acc;

    wire [OUTPUT_WIDTH-1:0] mul_res;
    wire [OUTPUT_WIDTH-1:0] add_res;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_reg   <= 0;
            weight_reg <= 0;
            product    <= 0;
            acc        <= 0;
            output_reg <= 0;
        end else begin
            // stage 1: latch inputs
            data_reg   <= data_in;
            weight_reg <= weight_in;

            // stage 2: multiply from registered inputs
            product    <= mul_res;  // Hold previous product for accumulation

            // stage 3: accumulate previous product
            acc   <= add_res; 

            // stage 4: latch to output
            output_reg <= acc;
        end
    end
assign forwarded_data_out=data_in;
assign forwarded_weight_out=weight_in;

add_fixed #(
    .WIDTH(DATA_WIDTH),
    .FRAC_BITS(6),
    .INT_BITS(5)
) adder_inst (
    .a(product),  // Truncate acc to DATA_WIDTH bits
    .b(acc), // Truncate product to DATA_WIDTH bits
    .sum(add_res), // Store result back in acc (truncated)
    .overflow()                // Ignore overflow for now
);
mul_fixed #(
    .WIDTH(DATA_WIDTH),
    .FRAC_BITS(6),
    .INT_BITS(5)
) multiplier_inst (
    .a(data_reg),
    .b(weight_reg),
    .prod(mul_res), // Truncate product to DATA_WIDTH bits
    .overflow()                      // Ignore overflow for now
);

endmodule

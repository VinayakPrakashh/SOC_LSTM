module processing_element #(
    parameter DATA_WIDTH = 12,
    parameter OUTPUT_WIDTH = 12
)(
    input  wire clk,
    input  wire rst,
    input  wire [DATA_WIDTH-1:0] data_in,
    input  wire [DATA_WIDTH-1:0] weight_in,
    output reg  [OUTPUT_WIDTH-1:0] output_reg,  // Changed back to reg
    output wire [DATA_WIDTH-1:0] forwarded_data_out
);

    wire [OUTPUT_WIDTH-1:0] mul_res;
    wire [OUTPUT_WIDTH-1:0] add_res;
    reg [OUTPUT_WIDTH-1:0] acc;  // Accumulator register

    // FIXED: Both accumulator and output are registered for proper timing
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc <= 0;
            output_reg <= 0;
        end else begin
            acc <= add_res;        // Update accumulator for next cycle
            output_reg <= add_res; // Output current result (registered)
        end
    end

    // Forward input immediately (combinational)
    assign forwarded_data_out = data_in;

    // Current multiplication
    mul_fixed #(
        .WIDTH(DATA_WIDTH),
        .FRAC_BITS(6),
        .INT_BITS(5)
    ) multiplier_inst (
        .a(data_in),    // Current inputs
        .b(weight_in),
        .prod(mul_res),
        .overflow()
    );

    // Current accumulation (previous acc + current multiply)
    add_fixed #(
        .WIDTH(DATA_WIDTH),
        .FRAC_BITS(6),
        .INT_BITS(5)
    ) adder_inst (
        .a(acc),        // Previous accumulator value
        .b(mul_res),    // Current multiplication result
        .sum(add_res),  // Current accumulate result
        .overflow()
    );

endmodule
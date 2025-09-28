// Sigmoid Address Calculator (Sign Independent)
// Converts input value magnitude in range [0, 6] to LUT address [0, 383]
// Input format: S1.5.6 (12-bit: 1 sign + 5 integer + 6 fractional bits)
// LUT covers range [0, 6] with 384 entries, so step = 6/384 = 0.015625
// Uses absolute value (magnitude) regardless of sign

module sigmoid_addr_calc #(
    parameter INPUT_WIDTH = 12,     // S1.5.6 format
    parameter FRAC_BITS = 6,        // Fractional bits
    parameter LUT_SIZE = 384,       // Number of LUT entries
    parameter ADDR_WIDTH = 9        // Address width (log2(384) = 9)
) (
    input [INPUT_WIDTH-1:0] data_in,
    output reg [ADDR_WIDTH-1:0] addr_out,
    output reg out_of_range
);

// LUT step size: 6.0 / 384 = 0.015625 = 1/64
// In S1.5.6 format: 0.015625 * 64 = 1
// So each LUT entry represents 1 LSB in fractional part

// Extract sign and magnitude (always use magnitude for address calculation)
wire sign_bit = data_in[INPUT_WIDTH-1];
wire [INPUT_WIDTH-2:0] magnitude = data_in[INPUT_WIDTH-2:0];

// Check for out of range conditions
wire [INPUT_WIDTH-2:0] max_value = 11'b00110_000000; // 6.0 in S1.5.6 format (magnitude part)

always @(*) begin
    // Use magnitude regardless of sign
    if (magnitude > max_value) begin
        // |Input| > 6.0 - clamp to maximum address
        addr_out = LUT_SIZE - 1; // 383
        out_of_range = 1'b1;
    end else begin
        // Valid range [0, 6] for magnitude
        // Direct mapping: magnitude itself is the scaled address
        // Since step = 1 LSB, and magnitude is in 1.5.6 format
        // We need to scale by 384/6 = 64 = 2^6
        // magnitude * 64 / 64 = magnitude (since step is 1 LSB)
        
        // For more precision, we could use: magnitude * 384 / (6 * 64)
        // But since 6 * 64 = 384, this simplifies to just magnitude
        // Limited to LUT_SIZE-1 to prevent overflow
        
        if (magnitude >= LUT_SIZE) begin
            addr_out = LUT_SIZE - 1;
            out_of_range = 1'b1;
        end else begin
            addr_out = magnitude[ADDR_WIDTH-1:0];
            out_of_range = 1'b0;
        end
    end
end

endmodule
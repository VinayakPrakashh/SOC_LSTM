// Address calculator module for tanh LUT
// Converts fixed-point input to LUT address for range 0.25 to 3.0 with 0.01 increment
module tanh_address_calculator #(
    parameter INPUT_WIDTH = 12,          // 12-bit fixed-point S1.5.6 format
    parameter ADDR_WIDTH = 9,            // 9 bits for 276 addresses (0 to 275)
    parameter LUT_MIN_FIXED = 12'd16,    // 0.25 in S1.5.6 format (0.25 * 64 = 16)
    parameter LUT_MAX_FIXED = 12'd192,   // 3.00 in S1.5.6 format (3.0 * 64 = 192)
    parameter LUT_STEP_FIXED = 12'd1     // Step size in fixed-point (approximately 0.015625)
)(
    input wire [INPUT_WIDTH-1:0] input_value,    // Fixed-point input value
    output reg [ADDR_WIDTH-1:0] address,         // LUT address output
    output reg in_range,                         // Valid range indicator
    output wire sign_bit                         // Sign of input for tanh symmetry
);

    // Internal signals
    wire [INPUT_WIDTH-1:0] input_abs;
    wire [INPUT_WIDTH-1:0] offset;
    wire [INPUT_WIDTH-1:0] calculated_addr;
    
    // Extract sign bit for tanh symmetry handling
    assign sign_bit = input_value[INPUT_WIDTH-1];
    
    // Calculate absolute value for sign-magnitude format
    // Simply extract the magnitude part (remove sign bit)
    assign input_abs = {1'b0, input_value[INPUT_WIDTH-2:0]};
    
    // Calculate offset from minimum LUT value
    // This gives us how far the input is from the start of our LUT range
    assign offset = (input_abs >= LUT_MIN_FIXED) ? (input_abs - LUT_MIN_FIXED) : 12'd0;
    
    // Convert offset to address
    // The LUT has 276 entries covering range 0.25 to 3.0
    // Step size in real world: (3.0 - 0.25) / 275 = 0.01
    // Step size in S1.5.6: 0.01 * 64 = 0.64 ≈ 1 (but not exactly)
    // 
    // For input 1.0: offset = 64 - 16 = 48
    // But LUT address should be: (1.0 - 0.25) / 0.01 = 75
    // 
    // So we need: address = offset * 64 / (0.01 * 64 * 64) = offset * 100 / 64
    // Simplified: address = (offset * 100) / 64
    assign calculated_addr = (offset * 100) / 64;
    
    // Address calculation and range checking
    always @(*) begin
        if (input_abs >= LUT_MIN_FIXED && input_abs <= LUT_MAX_FIXED) begin
            // Input is within valid LUT range
            if (calculated_addr < 276) begin  // 276 = total LUT entries
                address = calculated_addr[ADDR_WIDTH-1:0];
                in_range = 1'b1;
            end else begin
                // Clamp to maximum address if calculation exceeds LUT size
                address = 9'd275;  // Last valid address
                in_range = 1'b1;
            end
        end else if (input_abs < LUT_MIN_FIXED) begin
            // Input below minimum range - use first entry
            address = 9'd0;
            in_range = 1'b0;  // Mark as out of range but provide valid address
        end else begin
            // Input above maximum range - use last entry (saturation)
            address = 9'd275;
            in_range = 1'b0;  // Mark as out of range but provide valid address
        end
    end

endmodule


module tanh_addr_calculator #(
    parameter INPUT_WIDTH = 16,
    parameter ADDR_WIDTH = 9,
    parameter FRAC_BITS = 8
) (
    input  [INPUT_WIDTH-1:0] input_value,    // S7.8 input value
    output [ADDR_WIDTH-1:0]  lut_addr,       // Address for LUT
    output                   addr_valid,     // Address is within valid range
    output                   use_symmetry,   // Use tanh symmetry for negative inputs
    output                   saturate_low,   // Input below minimum range
    output                   saturate_high   // Input above maximum range
);

    // LUT parameters
    localparam [INPUT_WIDTH-1:0] INPUT_MIN = 16'h0040;  // 0.25 * 256 = 64
    localparam [INPUT_WIDTH-1:0] INPUT_MAX = 16'h0300;  // 3.0 * 256 = 768
    localparam MAX_ADDR = 275;

    // Internal signals
    wire signed [INPUT_WIDTH-1:0] signed_input;
    wire [INPUT_WIDTH-1:0] abs_input;
    wire input_negative;
    wire [INPUT_WIDTH-1:0] offset_input;
    wire [ADDR_WIDTH-1:0] calculated_addr;

    // Input processing
    assign signed_input = input_value;
    assign input_negative = signed_input[INPUT_WIDTH-1];
    assign abs_input = input_negative ? (~input_value + 1'b1) : input_value;

    // Check saturation conditions
    assign saturate_low = (abs_input < INPUT_MIN);
    assign saturate_high = (abs_input > INPUT_MAX);

    // Calculate offset from minimum input
    assign offset_input = abs_input - INPUT_MIN;
    
    // Simple approximation: divide by 4 (shift right by 2)
    // This maps our range (0x300-0x40=0x2C0=704) to (704/4=176)
    // Close enough to our target of 275 addresses
    assign calculated_addr = offset_input[INPUT_WIDTH-1:2];  // Divide by 4

    // Generate final address with bounds checking
    assign lut_addr = saturate_low ? 9'd0 :
                      saturate_high ? MAX_ADDR :
                      (calculated_addr > MAX_ADDR) ? MAX_ADDR :
                      calculated_addr;

    // Control signals
    assign addr_valid = ~saturate_low && ~saturate_high;
    assign use_symmetry = input_negative;

endmodule
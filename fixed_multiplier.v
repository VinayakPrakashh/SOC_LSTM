module mul_fixed #(
    parameter WIDTH      = 12, // Total bits (1 sign + integer + fraction)
    parameter FRAC_BITS  = 6,  // Number of fractional bits
    parameter INT_BITS   = 5   // Number of integer bits
)(
    input  [WIDTH-1:0] a,
    input  [WIDTH-1:0] b,
    output [WIDTH-1:0] prod,
    output overflow
);

    // Extract signs and magnitudes
    wire sign_a = a[WIDTH-1];
    wire sign_b = b[WIDTH-1];
    wire [WIDTH-2:0] mag_a = a[WIDTH-2:0];
    wire [WIDTH-2:0] mag_b = b[WIDTH-2:0];

    // Multiply unsigned magnitudes (double-width)
    wire [(2*(WIDTH-1))-1:0] full_prod = mag_a * mag_b;

    // Adjust for fractional bits
    wire [(2*(WIDTH-1))-1:0] shifted = full_prod >> FRAC_BITS;

    // Compute final sign (XOR of input signs)
    wire sign_out = sign_a ^ sign_b;

    // Maximum magnitude for output (11 bits for Q6.6 without sign)
    localparam [WIDTH-2:0] MAX_MAG = { (WIDTH-1){1'b1} }; // 11 ones

    // Candidate magnitude (truncate to WIDTH-1 bits)
    wire [WIDTH-2:0] mag_result = shifted[WIDTH-2:0];

    // Overflow detection: if shifted product exceeds MAX_MAG
    assign overflow = (shifted > MAX_MAG);

    // Saturate if overflow
    wire [WIDTH-2:0] sat_mag = overflow ? MAX_MAG : mag_result;

    // Zero case: if magnitude is zero, force sign to 0
    assign prod = { (sat_mag == 0) ? 1'b0 : sign_out, sat_mag };

endmodule

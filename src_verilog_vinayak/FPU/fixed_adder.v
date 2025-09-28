module add_fixed #(
    parameter WIDTH = 12,        // Total bits: 1 sign + 5 integer + 6 fraction
    parameter FRAC_BITS = 6,     // Number of fractional bits
    parameter INT_BITS  = 5      // Number of integer bits
)(
    input  [WIDTH-1:0] a,
    input  [WIDTH-1:0] b,
    output [WIDTH-1:0] sum,
    output overflow
);

    // Split sign and magnitude
    wire sign_a = a[WIDTH-1];
    wire sign_b = b[WIDTH-1];
    wire [WIDTH-2:0] mag_a = a[WIDTH-2:0];
    wire [WIDTH-2:0] mag_b = b[WIDTH-2:0];

    // Convert magnitudes to unsigned integers for arithmetic
    wire [WIDTH-1:0] ext_mag_a = {1'b0, mag_a};
    wire [WIDTH-1:0] ext_mag_b = {1'b0, mag_b};

    // Intermediate sum/diff
    reg [WIDTH:0] temp_mag;      // One extra bit for overflow on magnitude
    reg temp_sign;

    always @(*) begin
        if (sign_a == sign_b) begin
            // Same sign: add magnitudes
            temp_mag  = ext_mag_a + ext_mag_b;
            temp_sign = sign_a;
        end else begin
            // Different signs: subtract smaller magnitude from larger
            if (ext_mag_a >= ext_mag_b) begin
                temp_mag  = ext_mag_a - ext_mag_b;
                temp_sign = sign_a; // sign of the larger magnitude
            end else begin
                temp_mag  = ext_mag_b - ext_mag_a;
                temp_sign = sign_b;
            end
        end
    end

    // Maximum magnitude (without sign) = 5 integer bits + 6 fraction bits = 11 bits
    localparam [WIDTH-2:0] MAX_MAG = { (WIDTH-1){1'b1} }; // 11 ones

    // Overflow occurs if magnitude exceeds MAX_MAG after addition
    assign overflow = (temp_mag[WIDTH-1:0] > {1'b0, MAX_MAG});

    // Saturate on overflow
    wire [WIDTH-2:0] sat_mag = overflow ? MAX_MAG : temp_mag[WIDTH-2:0];

    // Construct output: sign + magnitude
    assign sum = { temp_mag == 0 ? 1'b0 : temp_sign, sat_mag };

endmodule

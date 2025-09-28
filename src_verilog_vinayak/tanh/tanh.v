`timescale 1ns / 1ps

module tanh_calc #(
    parameter WIDTH = 12,
    parameter FRAC_BITS = 6
) (
    input [WIDTH-1:0] in,
    output reg [WIDTH-1:0] out
);

    // Parameters for LUT
    parameter [WIDTH-1:0] LUT_MIN = 12'd16;    // 0.25 in S1.5.6
    parameter [WIDTH-1:0] LUT_MAX = 12'd192;   // 3.0 in S1.5.6
    parameter [WIDTH-1:0] NEG_LUT_MIN = 12'b100000010000;  // -0.25 in sign-magnitude
    parameter [WIDTH-1:0] NEG_LUT_MAX = 12'h8C0;          // -3.0 in sign-magnitude (12'd2240)


wire [8:0] address;
wire in_range,sign_bit;
wire [11:0] tanh_lut_out;
wire a_lt_b_min, a_gt_b_zero, a_eq_b, a_gte_b_min;
wire a_gt_b_neg_min, a_lt_b_neg_max, a_gt_b_max, a_lte_b_neg_min, a_lte_b_max, a_gt_b_neg_max;

always @(*) begin
    if( a_lt_b_min && a_gt_b_zero ) begin
        out = in; // tanh(x) ≈ x for small x
    end
    else if( a_gt_b_neg_min && a_lt_b_zero ) begin
        out = in; // tanh(x) ≈ x for small x
    end
    else if( a_gt_b_max ) begin
        out = 12'b000001000000; // tanh(x) ≈ 1 for large x, 1 in S1.5.6 is 0.111111 = 63
    end
    else if( a_lt_b_neg_max ) begin
        out = 12'b100001000000; // tanh(x) ≈ -1 for large negative x, -1 in S1.5.6 is 1.000001 = -63
    end
    else if( a_eq_b ) begin
        out = 0; // tanh(0) = 0
    end
    else if( a_gte_b_min && a_lte_b_max && a_gt_b_zero ) begin
        out = tanh_lut_out; // Use LUT output for positive range [0.25, 3.0]
    end
    else if(a_lte_b_neg_min && a_gt_b_neg_max && a_lt_b_zero) begin
        out = {1'b1, tanh_lut_out[10:0]}; // Flip sign bit to make LUT output negative
    end
    else begin
        out = 0; // Default case (should not occur)
    end
end

tanh_address_calculator addr_calc_inst (
    .input_value(in),
    .address(address),         // Connect to LUT address input
    .in_range(in_range),       // Can be used for additional logic if needed
    .sign_bit(sign_bit)        // Can be used for additional logic if needed
);
tanh_lut_ram lut_inst (
    .addr(address),          // Connect from addr_calc_inst.address
    .tanh_out(tanh_lut_out)    // Connect to output
);

fixed_point_comparator #(
    .WIDTH(12),
    .FRAC_BITS(6)
) comparator_inst (
    .a(in),
    .b(LUT_MIN),
    .a_gt_b(), .a_lt_b(a_lt_b_min), .a_eq_b(), .a_gte_b(a_gte_b_min), .a_lte_b()
);
fixed_point_comparator #(
    .WIDTH(12),
    .FRAC_BITS(6)
) comparator_inst_neg (
    .a(in),
    .b(12'b0),
    .a_gt_b(a_gt_b_zero), .a_lt_b(a_lt_b_zero), .a_eq_b(a_eq_b), .a_gte_b(), .a_lte_b()
);
fixed_point_comparator #(
    .WIDTH(12),
    .FRAC_BITS(6)
) comparator_inst_neg_min (
    .a(in),
    .b(NEG_LUT_MIN),
    .a_gt_b(a_gt_b_neg_min), .a_lt_b(), .a_eq_b(), .a_gte_b(), .a_lte_b(a_lte_b_neg_min)
);
fixed_point_comparator #(
    .WIDTH(12),
    .FRAC_BITS(6)
) comparator_inst_max (
    .a(in),
    .b(LUT_MAX),
    .a_gt_b(a_gt_b_max), .a_lt_b(), .a_eq_b(), .a_gte_b(), .a_lte_b(a_lte_b_max)
);
fixed_point_comparator #(
    .WIDTH(12),
    .FRAC_BITS(6)
) comparator_inst_neg_max (
    .a(in),
    .b(NEG_LUT_MAX),
    .a_gt_b(), .a_lt_b(a_lt_b_neg_max), .a_eq_b(), .a_gte_b(a_gt_b_neg_max), .a_lte_b()
);
endmodule

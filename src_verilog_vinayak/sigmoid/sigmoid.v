module sigmoid #(
    parameter WIDTH = 12,
    parameter FRAC_BITS = 6
) (
    input [WIDTH-1:0] in,
    output reg [WIDTH-1:0] out
);
    
parameter MAX_VAL = 12'b0_00110_000000; // 6.0 in Q6.6
parameter MIN_VAL = 12'b1_00110_000000; // -6.0 in Q6.6

wire [8:0] address; // 9 bits for 512 entries
wire in_range;
wire [WIDTH-1:0] lut_out;
wire [WIDTH-1:0] sub_out;
wire a_gt_b_max, a_lt_b_max, a_eq_b_zero, a_gte_b_zero, a_lte_b_max;
wire a_lt_b_zero, a_lt_b_neg_max, a_gte_b_neg_max;



always @(*) begin
    if(a_gt_b_max)begin
    out = 12'b0_00001_000000; // 1.0 in Q6.6
    end
    else if(a_gte_b_zero && a_lte_b_max) begin
        out = lut_out; // LUT output for range [0, 6]
    end
    else if(a_lt_b_neg_max) begin
        out = 0; // very small value close to 0
    end
    else if(a_lt_b_zero && a_gte_b_neg_max) begin
        out = sub_out; // 1 - LUT output for range [-6, 0]
    end
    else begin
        out = 12'b0; // very small value close to 0
    end
end
fixed_point_comparator #(
    .WIDTH(WIDTH),
    .FRAC_BITS(FRAC_BITS)
) comparator (
    .a(in),
    .b(MAX_VAL),
    .a_gt_b(a_gt_b_max),
    .a_lt_b(a_lt_b_max),
    .a_eq_b(),
    .a_gte_b(),
    .a_lte_b(a_lte_b_max)
);
fixed_point_comparator #(
    .WIDTH(WIDTH),
    .FRAC_BITS(FRAC_BITS)
) comparator_zero (
    .a(in),
    .b(12'b0),
    .a_gt_b(),
    .a_lt_b(a_lt_b_zero),
    .a_eq_b(a_eq_b_zero),
    .a_gte_b(a_gte_b_zero),
    .a_lte_b()
);
fixed_point_comparator #(
    .WIDTH(WIDTH),
    .FRAC_BITS(FRAC_BITS)
) comparator_neg (
    .a(in),
    .b(MIN_VAL), // -6.0 in Q6.6
    .a_gt_b(),
    .a_lt_b(a_lt_b_neg_max),
    .a_eq_b(),
    .a_gte_b(a_gte_b_neg_max),
    .a_lte_b()
);
sigmoid_addr_calc #(
    .INPUT_WIDTH(WIDTH),
    .FRAC_BITS(FRAC_BITS),
    .LUT_SIZE(384),
    .ADDR_WIDTH(9)
) addr_calc_inst (
    .data_in(in),
    .addr_out(address),
    .out_of_range(in_range)
);
sigmoid_lut lut_inst (
    .in(address), 
    .out(lut_out)
);
sub_fixed #(
    .WIDTH(WIDTH),
    .FRAC_BITS(FRAC_BITS)
) sub_inst (
    .a(12'b0_00001_000000), // 1.0 in Q6.6
    .b(lut_out),
    .diff(sub_out),
    .overflow()
);
endmodule

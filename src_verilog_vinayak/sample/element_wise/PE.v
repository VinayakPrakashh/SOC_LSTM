module PE #(
    parameter WIDTH = 12,
    parameter FRAC_BITS = 6,
    parameter INT_BITS = 5
) (

    input [WIDTH-1:0] register_o,
    input [WIDTH-1:0]  register_i,
    input [WIDTH-1:0]  register_g,
    input [WIDTH-1:0]  register_f,
    input [WIDTH-1:0]  register_c_prev,
    output [WIDTH-1:0] register_c,
    output [WIDTH-1:0] register_h
);

wire [WIDTH-1:0] mul_i_out;
wire [WIDTH-1:0] mul_f_out;
wire [WIDTH-1:0] tanh_out;



mul_fixed #(
    .WIDTH(WIDTH),
    .FRAC_BITS(FRAC_BITS),
    .INT_BITS(INT_BITS)
) mul_i (
    .a(register_i),
    .b(register_g),
    .prod(mul_i_out),
    .overflow()
);
mul_fixed #(
    .WIDTH(WIDTH),
    .FRAC_BITS(FRAC_BITS),
    .INT_BITS(INT_BITS)
) mul_f (
    .a(register_f),
    .b(register_c_prev),
    .prod(mul_f_out),
    .overflow()
);
add_fixed #(
    .WIDTH(WIDTH),
    .FRAC_BITS(FRAC_BITS),
    .INT_BITS(INT_BITS)
) add_c (
    .a(mul_f_out),
    .b(mul_i_out),
    .sum(register_c),
    .overflow()
);
tanh_calc #(
    .WIDTH(WIDTH),
    .FRAC_BITS(FRAC_BITS)
) tanh_inst (
    .in(register_c),
    .out(tanh_out)
);
mul_fixed #(
    .WIDTH(WIDTH),
    .FRAC_BITS(FRAC_BITS),
    .INT_BITS(INT_BITS)
) mul_h (
    .a(tanh_out),
    .b(register_o),
    .prod(register_h),
    .overflow()
);
endmodule
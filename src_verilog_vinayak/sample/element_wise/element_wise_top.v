module elem_wise_top #(
    parameter DATA_WIDTH = 12,
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input start,
    output done

);

wire [ADDRESS_BITS-1:0] o_addr_o;
wire [DATA_WIDTH-1:0] i_register_i, f_register_i, c_register_i, o_register_i;
wire [DATA_WIDTH-1:0] ct_minus_1;
wire [DATA_WIDTH-1:0] ct_output, ht_output;
wire we;


element_wise #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) elem_wise_inst (
    .clk(clk),
    .rst(rst),
    .start(start),
    .i_register_i(i_register_i),
    .f_register_i(f_register_i),
    .c_register_i(c_register_i),
    .o_register_i(o_register_i),
    .ct_minus_1(ct_minus_1),
    .o_addr_o(o_addr_o),
    .ct_output(ct_output),
    .ht_output(ht_output),
    .done(done),
    .we(we)
);
// Internal signals for buffer connections
buffer_i #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) buffer_i_inst (
    .clk(clk),
    .rst(rst),
    .we(1'b0),  // Read-only in this context
    .addr(o_addr_o),
    .din(0),    // No data input for read-only
    .dout(i_register_i)
);
buffer_f #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) buffer_f_inst (
    .clk(clk),
    .rst(rst),
    .we(1'b0),  // Read-only in this context
    .addr(o_addr_o),
    .din(0),    // No data input for read-only
    .dout(f_register_i)
);
buffer_c #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) buffer_c_inst (
    .clk(clk),
    .rst(rst),
    .we(1'b0),  // Read-only in this context
    .addr(o_addr_o),
    .din(0),    // No data input for read-only
    .dout(c_register_i)
);
buffer_o #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) buffer_o_inst (
    .clk(clk),
    .rst(rst),
    .we(1'b0),  // Read-only in this context
    .addr(o_addr_o),
    .din(0),    // No data input for read-only
    .dout(o_register_i)
);
buffer_ct #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) buffer_ct_inst (
    .clk(clk),
    .rst(rst),
    .we(we),  // Read-only in this context
    .addr(o_addr_o),   // Address not used for previous cell state
    .din(ct_output),    // No data input for read-only
    .dout(ct_minus_1)
);
buffer_ht #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) buffer_ht_inst (
    .clk(clk),
    .rst(rst),
    .we(we),  // Read-only in this context
    .addr(o_addr_o),   // Address not used for hidden state output
    .din(ht_output),    // No data input for read-only
    .dout()
);

endmodule

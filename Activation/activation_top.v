module lstm_activation_top #(
    parameter DATA_WIDTH = 12,
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input start,
    output done
);

// Internal signals for buffer connections
wire [ADDRESS_BITS-1:0] addr_i, addr_f, addr_c, addr_o;
wire [DATA_WIDTH-1:0] data_from_i, data_from_f, data_from_c, data_from_o;
wire [DATA_WIDTH-1:0] data_to_i, data_to_f, data_to_c, data_to_o;
wire we_i, we_f, we_c, we_o;

// Instantiate Input Gate Buffer
buffer_i input_buffer (
    .clk(clk),
    .rst(rst),
    .we(),
    .addr(addr_i),
    .din(data_to_i),
    .dout(data_from_i)
);

// Instantiate Forget Gate Buffer
buffer_f forget_buffer (
    .clk(clk),
    .rst(rst),
    .we(),
    .addr(addr_f),
    .din(data_to_f),
    .dout(data_from_f)
);

// Instantiate Cell Gate Buffer
buffer_c cell_buffer (
    .clk(clk),
    .rst(rst),
    .we(),
    .addr(addr_c),
    .din(data_to_c),
    .dout(data_from_c)
);

// Instantiate Output Gate Buffer
buffer_o output_buffer (
    .clk(clk),
    .rst(rst),
    .we(),
    .addr(addr_o),
    .din(data_to_o),
    .dout(data_from_o)
);

// Instantiate Activation Module
activate activation_unit (
    .clk(clk),
    .rst(rst),
    .start(start),
    
    // Input gate connections
    .in_data_i(data_from_i),
    .in_addr_i(addr_i),
    .out_data_i(data_to_i),
    .we_i(we_i),
    
    // Forget gate connections
    .in_data_f(data_from_f),
    .in_addr_f(addr_f),
    .out_data_f(data_to_f),
    .we_f(we_f),
    
    // Cell gate connections
    .in_data_c(data_from_c),
    .in_addr_c(addr_c),
    .out_data_c(data_to_c),
    .we_c(we_c),
    
    // Output gate connections
    .in_data_o(data_from_o),
    .in_addr_o(addr_o),
    .out_data_o(data_to_o),
    .we_o(we_o),
    
    .done(done)
);

endmodule
module lstm_activation_top #(
    parameter DATA_WIDTH = 16,      // Changed from 12 to 16 for S7.8
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input start,
    output done
);

// Internal signals for buffer connections
wire [ADDRESS_BITS-1:0] addr_i, addr_f, addr_c, addr_o;
wire [ADDRESS_BITS-1:0] raddr_i, raddr_f, raddr_c, raddr_o;  // Read addresses
wire [DATA_WIDTH-1:0] data_from_i, data_from_f, data_from_c, data_from_o;
wire [DATA_WIDTH-1:0] data_to_i, data_to_f, data_to_c, data_to_o;
wire we_i, we_f, we_c, we_o;
wire [ADDRESS_BITS-1:0] address;  // Address from activation unit

// Instantiate Input Gate Buffer (S7.8)
buffer_i #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) input_buffer (
    .clk(clk),
    .rst(rst),
    .we(we_i),              // Connect write enable
    .addr(addr_i),          // Write address
    .raddr(raddr_i),        // Read address  
    .din(data_to_i),        // Data to write
    .dout(data_from_i)      // Data to read
);

// Instantiate Forget Gate Buffer (S7.8)
buffer_f #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) forget_buffer (
    .clk(clk),
    .rst(rst),
    .we(we_f),              // Connect write enable
    .addr(addr_f),          // Write address
    .raddr(raddr_f),        // Read address
    .din(data_to_f),        // Data to write
    .dout(data_from_f)      // Data to read
);

// Instantiate Cell Gate Buffer (S7.8)
buffer_g #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) cell_buffer (
    .clk(clk),
    .rst(rst),
    .we(we_c),              // Connect write enable
    .addr(addr_c),          // Write address
    .raddr(raddr_c),        // Read address
    .din(data_to_c),        // Data to write
    .dout(data_from_c)      // Data to read
);

// Instantiate Output Gate Buffer (S7.8)
buffer_o #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) output_buffer (
    .clk(clk),
    .rst(rst),
    .we(we_o),              // Connect write enable
    .addr(addr_o),          // Write address
    .raddr(raddr_o),        // Read address
    .din(data_to_o),        // Data to write
    .dout(data_from_o)      // Data to read
);

// Connect read addresses to activation unit addresses
assign raddr_i = addr_i;
assign raddr_f = addr_f;
assign raddr_c = addr_c;
assign raddr_o = addr_o;

// Instantiate Activation Module (S7.8)
activate #(
    .DATA_WIDTH(DATA_WIDTH),
    .ADDRESS_BITS(ADDRESS_BITS)
) activation_unit (
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
    
    .address(address),      // Connect address output
    .done(done)
);

endmodule
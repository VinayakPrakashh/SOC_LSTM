module top  (
   input clk,
   input rst_n,
   input start,
   input done

);

wire [79:0] input_data; // 80-bit input data for each timestep
wire [15:0] weights;    // 16-bit weights read from memory
wire [15:0] output_data; // 16-bit output data from the LSTM controller
wire rd_input;        // Read enable for input buffer
wire [6:0] weight_addr; // Address for weight memory
wire [6:0] addr_counter; // Address counter for input/weight loading
wire wr_en;           // Write enable for output data
wire [6:0] load_addr_counter; // Counter for loading data into systolic array
wire done_multiply;   // Signal from systolic array indicating loading is done
wire load_data;       // Signal to load data for systolic array


lstm_sequence_controller lstm_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .input_data(input_data),
    .rd_input(rd_input),
    .weights(weights),
    .weight_addr(weight_addr),
    .output_data(output_data),
    .addr_counter(addr_counter),
    .wr_en(wr_en),
    .done(done),
    .busy(busy),
    .load_data(load_data),
    .load_addr_counter(load_addr_counter),
    .done_multiply(done_multiply)
);
input_buffer input_buf (
    .clk(clk),
    .rst(),
    .wr_en(),
    .wr_addr(),
    .wr_data(),
    .rd_en(rd_input),
    .rd_addr(0),
    .rd_data(input_data)
);
memory_94x16 ht_mem (
    .clk(clk),
    .rst(),
    .wr_en(),
    .wr_addr(),
    .wr_data(),
    .rd_en(1),
    .rd_addr(weight_addr),
    .rd_data(weights)
);
memory_100x16 data_mem (
    .clk(clk),
    .rst(),
    .wr_en(wr_en),
    .wr_addr(addr_counter),
    .wr_data(output_data),
    .rd_en(),
    .rd_addr(load_addr_counter),
    .rd_data(mul_rd_data)
);

top_muliplier #(
    .DATA_WIDTH(16),
    .OUTPUT_WIDTH(16),
    .ADDR_WIDTH(16),
    .TILE_ADDR_WIDTH(3),
    .MATRIX_ROWS(376),
    .WEIGHT_MEM_SIZE(37600),
    .DATA_MEM_SIZE(94),
    .MATRIX_COLS(100),
    .DATA_TILE_WIDTH(2),
    .DATA_VECTOR_LENGTH(100),
    .TILE_WIDTH(4)
) systolic_array (
    .clk(clk),
    .rst_n(rst_n),
    .we(load_data), // Write enable when loading data
    .data_in(mul_rd_data), // Assuming we load one input at a time
    .weight_in(weights), // Weights for the current operation
    .gb_data_addr(), // Address for loading data into the systolic array
    .ht_output() // Output from the systolic array (not connected here)
);

endmodule

module top_s3_20 (
   input clk,
   input rst_n,
   input start,
   output done,
   output [23:0] final_output
);

    // Width parameter for S3.20 format
    localparam DATA_WIDTH = 24;  // 1 sign + 3 int + 20 frac bits
    
    // Internal signals - all widened to 24 bits
    wire [119:0] input_data;     // 120-bit input data (5 × 24-bit values per timestep)
    wire [23:0] weights;         // 24-bit weights read from memory
    wire [23:0] output_data;     // 24-bit output data from LSTM controller
    wire rd_input;               // Read enable for input buffer
    wire [6:0] weight_addr;      // Address for weight memory
    wire [6:0] addr_counter;     // Address counter for input/weight loading
    wire wr_en;                  // Write enable for output data
    wire [6:0] load_addr_counter; // Counter for loading data into systolic array
    wire done_multiply;          // Signal from systolic array indicating loading is done
    wire load_data;              // Signal to load data for systolic array
    wire [23:0] mul_rd_data;     // 24-bit data read from memory
    wire [6:0] ht_counter;       // Counter for hidden state updates
    wire [23:0] ht_output;       // 24-bit output from hidden state memory to multiplier
    wire inter_rst;
    wire [6:0] final_addr_ht;    // Final address for hidden state memory access
    wire start_fc_layer;         // Signal to start FC layer processing
    wire fc_done;                // Signal from FC layer indicating completion
    wire sel_fc;                 // Select signal for mux
    wire [6:0] addr_fc;          // Address for FC layer weight/bias access
    wire [23:0] ht_data;         // 24-bit data read from hidden state memory
    wire busy;

    // ========================================================================
    // LSTM SEQUENCE CONTROLLER (24-bit)
    // ========================================================================
    lstm_sequence_controller  lstm_ctrl (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .input_data(input_data),
        .rd_input(rd_input),
        .weights(ht_data),
        .weight_addr(weight_addr),
        .output_data(output_data),
        .addr_counter(addr_counter),
        .wr_en(wr_en),
        .done(done),
        .busy(busy),
        .load_data(load_data),
        .load_addr_counter(load_addr_counter),
        .done_multiply(done_multiply),
        .ht_counter(ht_counter),
        .inter_rst(inter_rst),
        .start_fc_layer(start_fc_layer),
        .fc_done(fc_done),
        .sel_fc(sel_fc)
    );

    // ========================================================================
    // INPUT BUFFER (120-bit width for 5 × 24-bit inputs)
    // ========================================================================
    sync_fifo #(
        .DATA_WIDTH(120),        // 5 × 24-bit values
        .DEPTH(20),
        .ADDR_WIDTH(5),
        .INIT_COUNT(20)
    ) input_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(1'b0),           // Connect to input data source
        .rd_en(rd_input),
        .wr_data(120'b0),       // Connect to input data source
        .rd_data(input_data),
        .full(),
        .empty()
    );

    // ========================================================================
    // HIDDEN STATE MEMORY (94 × 24-bit)
    // ========================================================================
    memory_94x24 ht_mem (
        .clk(clk),
        .rst(rst_n),
        .wr_en(done_multiply),
        .wr_addr(ht_counter),
        .wr_data(ht_output),
        .rd_en(1'b1),
        .rd_addr(final_addr_ht),
        .rd_data(ht_data)
    );

    // ========================================================================
    // DATA MEMORY (100 × 24-bit)
    // ========================================================================
    memory_100x24 data_mem (
        .clk(clk),
        .rst(rst_n),
        .wr_en(wr_en),
        .wr_addr(addr_counter),
        .wr_data(output_data),
        .rd_en(1'b1),
        .rd_addr(load_addr_counter),
        .rd_data(mul_rd_data)
    );

    // ========================================================================
    // SYSTOLIC ARRAY MULTIPLIER (24-bit)
    // ========================================================================
    top_multiplier #(
        .DATA_WIDTH(24),         // Changed from 16 to 24
        .OUTPUT_WIDTH(24),       // Changed from 16 to 24
        .ADDR_WIDTH(16),
        .TILE_ADDR_WIDTH(3),
        .MATRIX_ROWS(376),
        .WEIGHT_MEM_SIZE(37600),
        .DATA_MEM_SIZE(94),
        .MATRIX_COLS(100),
        .DATA_TILE_WIDTH(2),
        .DATA_VECTOR_LENGTH(100),
        .TILE_WIDTH(4)
    ) multiplier (
        .clk(clk),
        .rst_n(rst_n),
        .we(load_data),
        .data_in(mul_rd_data),
        .gb_data_addr(load_addr_counter),
        .ht_output(ht_output),
        .done_data(done_multiply),
        .inter_rst(inter_rst)
    );

    // ========================================================================
    // FULLY CONNECTED LAYER (24-bit)
    // ========================================================================
    fc_pe #(
        .DATA_WIDTH(24),         // Changed from 16 to 24
        .WEIGHT_WIDTH(24),       // Changed from 16 to 24
        .BIAS_WIDTH(24),         // Changed from 16 to 24
        .OUTPUT_WIDTH(24),       // Changed from 16 to 24
        .HIDDEN_SIZE(94)
    ) fc_processor (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_fc_layer),
        .ht_in(ht_data),
        .weight_in(weights),
        .bias_in(24'h01D7F7),    // Example bias: 0.000029 in S3.20 (scaled from 16'h001D)
        .addr(addr_fc),
        .done(fc_done),
        .fc_out(final_output)
    );

    // ========================================================================
    // FC WEIGHT MEMORY (94 × 24-bit)
    // ========================================================================
    fc_weight_bram #(
        .DATA_WIDTH(24),         // Changed from 16 to 24
        .ADDR_WIDTH(7),
        .MEM_SIZE(94)
    ) weight_bram (
        .clk(clk),
        .rst_n(rst_n),
        .addr(addr_fc),
        .dout(weights)
    );

    // ========================================================================
    // ADDRESS MUX (24-bit addresses, but only 7 bits used)
    // ========================================================================
    mux_2_1 mux (
        .a(weight_addr),
        .b(addr_fc),
        .sel(sel_fc),
        .out(final_addr_ht)
    );

endmodule
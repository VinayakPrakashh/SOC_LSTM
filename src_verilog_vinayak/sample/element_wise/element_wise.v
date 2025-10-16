module element_wise #(
    parameter DATA_WIDTH = 12,
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input start,
    
    // Activated gate values from buffers
    input [DATA_WIDTH-1:0] i_register_i,  // Input gate (i)
    input [DATA_WIDTH-1:0] f_register_i,  // Forget gate (f)
    input [DATA_WIDTH-1:0] c_register_i,  // Cell gate (g)
    input [DATA_WIDTH-1:0] o_register_i,  // Output gate (o)
    
    // Previous cell state input
    input [DATA_WIDTH-1:0] ct_minus_1,    // C(t-1)
    
    // Address outputs to read from buffers
    output reg [ADDRESS_BITS-1:0] o_addr_o,
    
    // LSTM outputs
    output reg [DATA_WIDTH-1:0] ct_output,    // New cell state C(t)
    output reg [DATA_WIDTH-1:0] ht_output,    // Hidden state h(t)
    output reg done,
    output reg we
);

// FSM states - Updated to 3 bits for 5 states
localparam IDLE = 3'b000;
localparam READ = 3'b001;
localparam COMPUTE = 3'b010;
localparam WRITE_BACK = 3'b011;
localparam DONE = 3'b100;

reg [2:0] state, next_state;  // Updated to 3 bits
reg [ADDRESS_BITS-1:0] counter;

// Internal registers for computation
reg [DATA_WIDTH-1:0] i_val, f_val, g_val, o_val, ct_prev;

// Wires for PE unit outputs
wire [DATA_WIDTH-1:0] ct_output_inter, ht_output_inter;

// State transitions
always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        counter <= 0;
    end else begin
        state <= next_state;
    end
end

// Next state logic
always @(*) begin
    case (state)
        IDLE: next_state = start ? READ : IDLE;
        READ: next_state = COMPUTE;
        COMPUTE: next_state = WRITE_BACK;
        WRITE_BACK: next_state = (counter == 2'b11) ? DONE : READ;  // Check after write_back
        DONE: next_state = IDLE;
        default: next_state = IDLE;
    endcase
end

// Main computation logic
always @(posedge clk or posedge rst) begin
    if (rst) begin
        o_addr_o <= 0; 
        ct_output <= 0; ht_output <= 0;
        i_val <= 0; f_val <= 0; g_val <= 0; o_val <= 0; ct_prev <= 0;
        done <= 0;
        we <= 0;
        counter <= 0;
    end else begin
        case (state)
            IDLE: begin
                done <= 0;
                counter <= 0;
                ct_prev <= ct_minus_1; // Store previous cell state
            end
            
            READ: begin
                we <=0; // Disable write
                // Set addresses to read from activated buffers
                o_addr_o <= counter;
            end
            
            COMPUTE: begin
                // Register the activated values
                i_val <= i_register_i;  // Input gate
                f_val <= f_register_i;  // Forget gate
                g_val <= c_register_i;  // Cell candidate
                o_val <= o_register_i;  // Output gate
                // PE unit computes in this cycle
            end
            
            WRITE_BACK: begin
                // Output the computed values
                ct_output <= ct_output_inter;
                ht_output <= ht_output_inter;
                we <=1'b1; // Enable write if needed
                counter <= counter + 1;
                // Counter is used for output addressing if needed
            end
            
            DONE: begin
                we <=1'b0; // Disable write
                done <= 1;
            end
        endcase
    end
end

// PE unit instantiation
PE #(
    .WIDTH(DATA_WIDTH),
    .FRAC_BITS(6),
    .INT_BITS(5)
) pe_unit (
    .register_o(o_val),
    .register_i(i_val),
    .register_g(g_val),
    .register_f(f_val),
    .register_c_prev(ct_prev),
    .register_c(ct_output_inter),
    .register_h(ht_output_inter)
);

endmodule
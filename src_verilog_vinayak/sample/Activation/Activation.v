module activate #(
    parameter DATA_WIDTH = 12,
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input start,
    // input gate
    input [DATA_WIDTH-1:0] in_data_i,
    output reg [ADDRESS_BITS-1:0] in_addr_i,
    output reg [DATA_WIDTH-1:0] out_data_i,
    output reg we_i,
    // forget gate
    input [DATA_WIDTH-1:0] in_data_f,
    output reg [ADDRESS_BITS-1:0] in_addr_f,
    output reg [DATA_WIDTH-1:0] out_data_f,
    output reg we_f,
    // candidate gate
    input [DATA_WIDTH-1:0] in_data_c,
    output reg [ADDRESS_BITS-1:0] in_addr_c,
    output reg [DATA_WIDTH-1:0] out_data_c,
    output reg we_c,
    // output gate
    input [DATA_WIDTH-1:0] in_data_o,
    output reg [ADDRESS_BITS-1:0] in_addr_o,
    output reg [DATA_WIDTH-1:0] out_data_o,
    output reg we_o,
    
    output reg done
);

// Modified FSM states for BRAM timing
localparam IDLE = 3'b000;
localparam READ = 3'b001;     // Set address, wait for BRAM
localparam ACTIVATE = 3'b010; // Process valid data from BRAM
localparam DONE = 3'b011;

reg [2:0] state, next_state;
reg [ADDRESS_BITS-1:0] counter;

// Activation function outputs
wire [DATA_WIDTH-1:0] sigmoid_out_i, sigmoid_out_f, sigmoid_out_o;
wire [DATA_WIDTH-1:0] tanh_out_c;

// Instantiate activation functions
sigmoid #(.WIDTH(DATA_WIDTH), .FRAC_BITS(6)) sig_i (
    .in(in_data_i),
    .out(sigmoid_out_i)
);

sigmoid #(.WIDTH(DATA_WIDTH), .FRAC_BITS(6)) sig_f (
    .in(in_data_f),
    .out(sigmoid_out_f)
);

sigmoid #(.WIDTH(DATA_WIDTH), .FRAC_BITS(6)) sig_o (
    .in(in_data_o),
    .out(sigmoid_out_o)
);

tanh_calc #(.WIDTH(DATA_WIDTH), .FRAC_BITS(6)) tanh_c (
    .in(in_data_c),
    .out(tanh_out_c)
);

// State transitions
always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        counter <= 0;
    end else begin
        state <= next_state;
        
        // Counter management
        if (state == IDLE && start) begin
            counter <= 0;  // Reset counter when starting
        end else if (state == ACTIVATE) begin
            counter <= counter + 1;  // Increment after processing
        end
    end
end

// Next state logic
always @(*) begin
    case (state)
        IDLE: begin
            if (start)
                next_state = READ;
            else
                next_state = IDLE;
        end
        
        READ: begin
            // Always go to ACTIVATE after setting address
            next_state = ACTIVATE;
        end
        
        ACTIVATE: begin
            // Check if we've processed all 4 elements (0,1,2,3)
            if (counter == 2'b11)  // After processing element 3
                next_state = DONE;
            else
                next_state = READ;  // Read next element
        end
        
        DONE: begin
            next_state = IDLE;
        end
        
        default: next_state = IDLE;
    endcase
end

// Output logic
always @(posedge clk or posedge rst) begin
    if (rst) begin
        in_addr_i <= 0; in_addr_f <= 0; in_addr_c <= 0; in_addr_o <= 0;
        out_data_i <= 0; out_data_f <= 0; out_data_c <= 0; out_data_o <= 0;
        we_i <= 0; we_f <= 0; we_c <= 0; we_o <= 0;
        done <= 0;
    end else begin
        case (state)
            IDLE: begin
                we_i <= 0; we_f <= 0; we_c <= 0; we_o <= 0;
                done <= 0;
            end
            
            READ: begin
                // Set BRAM read addresses
                in_addr_i <= counter;
                in_addr_f <= counter;
                in_addr_c <= counter;
                in_addr_o <= counter;
                
                // No write during read
                we_i <= 0; we_f <= 0; we_c <= 0; we_o <= 0;
            end
            
            ACTIVATE: begin
                // Data is now valid from BRAM, apply activations
                out_data_i <= sigmoid_out_i;
                out_data_f <= sigmoid_out_f;
                out_data_c <= tanh_out_c;
                out_data_o <= sigmoid_out_o;
                
                // Keep addresses stable
                // (addresses were set in READ state)
            
                we_i <= 1; we_f <= 1; we_c <= 1; we_o <= 1;
            end
            
            DONE: begin
                we_i <= 0; we_f <= 0; we_c <= 0; we_o <= 0;
                done <= 1;
            end
        endcase
    end
end

endmodule
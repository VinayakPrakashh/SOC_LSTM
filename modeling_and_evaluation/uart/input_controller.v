`timescale 1ns /1ps

module input_controller #(
    parameter OUTPUT_DATA_WIDTH = 80,
    parameter ADDR_WIDTH = 5,
    parameter INPUT_DATA_WIDTH = 8
)(
    input clk,
    input start,
    input rst_n,
    input [INPUT_DATA_WIDTH-1:0] input_data,

    output reg pop_fifo,
    output [ADDR_WIDTH-1:0] output_addr,
    output reg [OUTPUT_DATA_WIDTH-1:0] output_data,
    output reg done,
    output reg wr_en,
    output reg start_lstm
);

assign output_addr = 0;

// FSM States
localparam IDLE         = 3'd0;
localparam POP          = 3'd1;
localparam WAIT_DATA    = 3'd2;
localparam OUTPUT_READY = 3'd3;
localparam START_LSTM = 3'd4;

reg [2:0] state, next_state;

reg [3:0] pop_counter;
reg [79:0] combined_data;


//------------------------------------------------
// State register
//------------------------------------------------
always @(posedge clk or negedge rst_n)
begin
    if(!rst_n)
        state <= IDLE;
    else
        state <= next_state;
end


//------------------------------------------------
// Next state logic
//------------------------------------------------
always @(*)
begin
    next_state = state;

    case(state)

    IDLE:
        if(start)
            next_state = POP;

    POP:
        next_state = WAIT_DATA;

    WAIT_DATA:
        if(pop_counter == 9)
            next_state = OUTPUT_READY;
        else
            next_state = POP;

    OUTPUT_READY:
    
        next_state = START_LSTM;

    START_LSTM:
        next_state = IDLE;

    default: next_state = IDLE;

    endcase
end


//------------------------------------------------
// Main logic
//------------------------------------------------
always @(posedge clk or negedge rst_n)
begin
    if(!rst_n)
    begin
        pop_counter  <= 0;
        pop_fifo     <= 0;
        combined_data <= 0;
        output_data  <= 0;
        done         <= 0;
        wr_en        <= 0;
        start_lstm   <= 0;
    end
    else
    begin
        case(state)

        //----------------------------------
        // IDLE
        //----------------------------------
        IDLE:
        begin
            pop_counter <= 0;
            pop_fifo    <= 0;
            done        <= 0;
            wr_en       <= 0;
            combined_data <= 0;
            output_data <= 0;
            start_lstm  <= 0;
        end

        //----------------------------------
        // POP FIFO (1-cycle read pulse)
        //----------------------------------
        POP:
        begin
            pop_fifo <= 1;
        end

        //----------------------------------
        // WAIT_DATA (FIFO output valid)
        //----------------------------------
        WAIT_DATA:
        begin
            pop_fifo <= 0;

            case(pop_counter)
                4'd0: combined_data[7:0]   <= input_data;
                4'd1: combined_data[15:8]  <= input_data;
                4'd2: combined_data[23:16] <= input_data;
                4'd3: combined_data[31:24] <= input_data;
                4'd4: combined_data[39:32] <= input_data;
                4'd5: combined_data[47:40] <= input_data;
                4'd6: combined_data[55:48] <= input_data;
                4'd7: combined_data[63:56] <= input_data;
                4'd8: combined_data[71:64] <= input_data;
                4'd9: combined_data[79:72] <= input_data;
            endcase

            pop_counter <= pop_counter + 1;
        end

        //----------------------------------
        // OUTPUT READY
        //----------------------------------
        OUTPUT_READY:
        begin
            output_data <= combined_data;
            wr_en       <= 1;
            done        <= 1;
        end
        START_LSTM: begin
            start_lstm <= 1;
        end

        endcase
    end
end

endmodule
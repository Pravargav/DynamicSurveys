import React from "react";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import ButtonGroup from "react-bootstrap/ButtonGroup";
import { useNavigate } from "react-router-dom";

function Nextform() {
    let navigate = useNavigate();

    const handleClick2 = async () => {
      navigate("/nextForm2");
    };

    const handleClick = async () => {
        navigate("/");
      };
  return (
    <div>
        <div
        style={{
          justifyContent: "center",
          alignItems: "center",
          height: "15vh",
          width: "20vh 20vh",
          border: "1px solid gray",
          margin: "70px 70px 70px 70px",
        }}
      >
        <Form>
          <h3 style={{ fontFamily: "sans-serif", textAlign: "center" }}>
            Do you want test accurately about your Problem??
          </h3>
          <ButtonGroup
            aria-label="Basic example"
            type="submit"
            size="lg"
            style={{ fontFamily: "sans-serif" }}
            className="w-100"
          >
            <Button onClick={handleClick2}>Yes</Button>
            <Button onClick={handleClick}>No</Button>
          </ButtonGroup>
        </Form>
      </div>
    </div>
  )
}

export default Nextform

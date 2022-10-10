/*
 *    This file is part of ACADO Toolkit.
 *
 *    ACADO Toolkit -- A Toolkit for Automatic Control and Dynamic Optimization.
 *    Copyright (C) 2008-2014 by Boris Houska, Hans Joachim Ferreau,
 *    Milan Vukov, Rien Quirynen, KU Leuven.
 *    Developed within the Optimization in Engineering Center (OPTEC)
 *    under supervision of Moritz Diehl. All rights reserved.
 *
 *    ACADO Toolkit is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with ACADO Toolkit; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/**
 *    \file   examples/simulation_environment/simple_mpc.cpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date   2009
 */

#include <math.h>

#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>

using namespace std;

USING_NAMESPACE_ACADO

int main()
{
	// INTRODUCE THE VARIABLES:
	// -------------------------
	DifferentialState position_x;
	DifferentialState position_y;
	DifferentialState position_z;
	DifferentialState velocity_x;
	DifferentialState velocity_y;
	DifferentialState velocity_z;
	DifferentialState roll;
	DifferentialState pitch;
	DifferentialState yaw;

	Control roll_ref;
	Control pitch_ref;
	Control thrust;

	/// TODO: Switch to online data
	// OnlineData roll_tau;
	// OnlineData roll_gain;
	// OnlineData pitch_tau;
	// OnlineData pitch_gain;
	// OnlineData yaw_rate_command;
	// OnlineData drag_coefficient_x;
	// OnlineData drag_coefficient_y;
	// OnlineData drag_coefficient_z;
	// OnlineData external_disturbances_x;
	// OnlineData external_disturbances_y;
	// OnlineData external_disturbances_z;

	double roll_tau{0.1};
	double roll_gain{1.0};
	double pitch_tau{0.1};
	double pitch_gain{1.0};
	double yaw_rate_command{0.0};

	// DEFINE A DIFFERENTIAL EQUATION:
	// -------------------------------
	DifferentialEquation f;

	f << dot(position_x) == velocity_x;
	f << dot(position_y) == velocity_y;
	f << dot(position_z) == velocity_z;
	f << dot(velocity_x) == (cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll)) * thrust; // - drag_coefficient_x * velocity_x + external_disturbances_x
	f << dot(velocity_y) == (sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll)) * thrust; // - drag_coefficient_y * velocity_y + external_disturbances_y
	f << dot(velocity_z) == cos(pitch) * cos(roll) * thrust - 9.81;
	f << dot(roll) == (1 / roll_tau) * (roll_gain * roll_ref - roll);
	f << dot(pitch) == (1 / pitch_tau) * (pitch_gain * pitch_ref - pitch);
	f << dot(yaw) == yaw_rate_command;

	// DEFINE LEAST SQUARE FUNCTION:
	// -----------------------------
	Function h;

	h << position_x;
	h << position_y;
	h << position_z;
	h << velocity_x;
	h << velocity_y;
	h << velocity_z;
	h << roll;
	h << pitch;
	h << roll_ref;
	h << cos(pitch) * cos(roll) * thrust - 9.81;

	DMatrix Q(10, 10);
	Q.setIdentity();
	Q(0, 0) = 10.0;
	Q(1, 1) = 10.0;
	Q(1, 1) = 10.0;

	Function hN;
	hN << position_x;
	hN << position_y;
	hN << position_z;
	hN << velocity_x;
	hN << velocity_y;
	hN << velocity_z;

	DMatrix QN(6, 6);
	QN.setIdentity();

	DVector r(10);
	r.setAll(0.0);

	// DEFINE AN OPTIMAL CONTROL PROBLEM:
	// ----------------------------------
	const double t_start = 0.0;
	const double t_end = 1.0;

	OCP ocp(t_start, t_end, 20);

	ocp.minimizeLSQ(Q, h, r);
	ocp.minimizeLSQEndTerm(QN, hN);
	ocp.subjectTo(f);

	ocp.subjectTo(-M_PI_4 <= roll_ref <= M_PI_4);
	ocp.subjectTo(-M_PI_4 <= pitch_ref <= M_PI_4);
	ocp.subjectTo(0.5 * 9.81 <= thrust <= 1.5 * 9.81);

	// SETTING UP THE (SIMULATED) PROCESS:
	// -----------------------------------
	OutputFcn identity;
	DynamicSystem dynamicSystem(f, identity);

	Process process(dynamicSystem, INT_RK45);

	// SETTING UP THE MPC CONTROLLER:
	// ------------------------------
	RealTimeAlgorithm alg(ocp, 0.05);
	alg.set(MAX_NUM_ITERATIONS, 2);

	StaticReferenceTrajectory zeroReference;

	Controller controller(alg, zeroReference);

	// SETTING UP THE SIMULATION ENVIRONMENT,  RUN THE EXAMPLE...
	// ----------------------------------------------------------
	SimulationEnvironment sim(0.0, 3.0, process, controller);

	DVector x0(9);
	x0(0) = 1.0;
	x0(1) = 0.0;
	x0(2) = 0.0;
	x0(3) = 0.0;
	x0(4) = 0.0;
	x0(5) = 0.0;
	x0(6) = 0.0;
	x0(7) = 0.0;
	x0(8) = 0.0;

	if (sim.init(x0) != SUCCESSFUL_RETURN)
		exit(EXIT_FAILURE);
	if (sim.run() != SUCCESSFUL_RETURN)
		exit(EXIT_FAILURE);

	// ...AND PLOT THE RESULTS
	// ----------------------------------------------------------
	VariablesGrid sampledProcessOutput;
	sim.getSampledProcessOutput(sampledProcessOutput);

	VariablesGrid feedbackControl;
	sim.getFeedbackControl(feedbackControl);

	GnuplotWindow window;
	window.addSubplot(sampledProcessOutput(0), "X [m]");
	window.addSubplot(sampledProcessOutput(1), "Y [m]");
	window.addSubplot(sampledProcessOutput(2), "Z [m]");
	window.addSubplot(sampledProcessOutput(3), "X Velocity [m/s]");
	window.addSubplot(sampledProcessOutput(4), "Y Velocity [m/s]");
	window.addSubplot(sampledProcessOutput(5), "Z Velocity [m/s]");
	window.addSubplot(sampledProcessOutput(6), "Roll");
	window.addSubplot(sampledProcessOutput(7), "Pitch");
	window.addSubplot(sampledProcessOutput(8), "Yaw");
	// window.addSubplot( feedbackControl(1),      "Damping Force [N]" );
	// window.addSubplot( feedbackControl(0),      "Road Excitation [m]" );
	window.plot();

	return EXIT_SUCCESS;
}
